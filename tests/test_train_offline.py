"""Runs the training step on a tiny randomly initialised Whisper, so the
collator, label handling and LoRA targets are checked without downloading
a checkpoint."""
import json

import numpy as np
import pytest
import soundfile as sf
import torch

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from telasr.augment import Augmenter, TelConfig
from telasr.train import LORA_TARGETS, Collator

START, EOS, PAD = 5, 6, 6


class CharTokenizer:
    """Stands in for WhisperTokenizer: <|startoftranscript|> + chars + eos."""

    def convert_tokens_to_ids(self, tok):
        return START

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        ids = [[START] + [10 + (ord(c) % 90) for c in t][: max_length - 2] + [EOS] for t in texts]
        n = max(map(len, ids))
        input_ids = torch.tensor([x + [PAD] * (n - len(x)) for x in ids])
        mask = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x in ids])
        return type("B", (), {"input_ids": input_ids, "attention_mask": mask})()


class Proc:
    def __init__(self):
        self.feature_extractor = transformers.WhisperFeatureExtractor()
        self.tokenizer = CharTokenizer()


def tiny_whisper():
    cfg = transformers.WhisperConfig(
        vocab_size=100, d_model=32, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=64, decoder_ffn_dim=64, max_source_positions=1500,
        max_target_positions=64, decoder_start_token_id=START, pad_token_id=PAD, eos_token_id=EOS,
    )
    return transformers.WhisperForConditionalGeneration(cfg)


def test_collator_and_lora_step(tmp_path):
    rows = []
    for i in range(4):
        p = tmp_path / f"{i}.wav"
        sf.write(p, np.random.default_rng(i).normal(0, 0.1, 16000 + 4000 * i).astype("float32"), 16000)
        rows.append({"id": str(i), "audio": str(p), "text": "नमस्ते दुनिया"[: 4 + i]})

    collate = Collator(Proc(), Augmenter([], TelConfig(p_amr=0.0)), seed=0)
    feats, labels = collate(rows)
    assert feats.shape == (4, 80, 3000)
    assert (labels[:, 0] != START).all()  # start token stripped
    assert (labels == -100).any()  # padding masked

    model = peft.get_peft_model(tiny_whisper(), peft.LoraConfig(r=4, lora_alpha=8, target_modules=LORA_TARGETS))
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora" in n for n in trainable)
    for t in ("q_proj", "fc1", "fc2"):
        assert any(t in n for n in trainable)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    losses = []
    for _ in range(5):
        loss = model(input_features=feats, labels=labels).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]

    model.save_pretrained(tmp_path / "adapter")
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    cfg = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert set(cfg["target_modules"]) == set(LORA_TARGETS)
