"""LoRA fine-tuning of Whisper on Hindi, with three data arms:

    clean    real training audio as is
    tel      real audio with on-the-fly phone-line degradation and noise
    tel+tts  as tel, plus round-trip-filtered synthetic speech mixed in at
             --tts-frac of each batch

All three arms use the same steps, learning rate, LoRA config and seed. The
only thing that changes is the data, so a WER difference between arms is
about the data.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .augment import Augmenter, TelConfig

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


def read_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class Collator:
    def __init__(self, processor, augment: Augmenter | None, seed: int, max_label_len: int = 200):
        self.proc, self.aug = processor, augment
        self.rng = np.random.default_rng(seed)
        self.max_label_len = max_label_len
        self.start = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    def __call__(self, rows: list[dict]):
        import soundfile as sf

        audio = []
        for r in rows:
            x, _ = sf.read(r["audio"], dtype="float32")
            if self.aug is not None and not r.get("no_aug"):
                x = self.aug(x, self.rng)
            audio.append(x)
        feats = self.proc.feature_extractor(audio, sampling_rate=16_000, return_tensors="pt").input_features
        lab = self.proc.tokenizer([r["text"] for r in rows], padding=True, truncation=True, max_length=self.max_label_len, return_tensors="pt")
        labels = lab.input_ids.masked_fill(lab.attention_mask.eq(0), -100)
        # The model prepends <|startoftranscript|> when it shifts labels right,
        # so drop it from the labels if the tokenizer already added it.
        if (labels[:, 0] == self.start).all():
            labels = labels[:, 1:]
        return feats, labels


def batches(real: list[dict], synth: list[dict], batch_size: int, tts_frac: float, rng: random.Random):
    n_syn = round(batch_size * tts_frac) if synth else 0
    while True:
        b = rng.sample(real, batch_size - n_syn) + (rng.sample(synth, n_syn) if n_syn else [])
        rng.shuffle(b)
        yield b


def train(
    arm: str,
    train_manifest: str,
    out_dir: str,
    tts_manifest: str | None = None,
    noise_files: list[str] | None = None,
    model_id: str = "openai/whisper-small",
    steps: int = 2000,
    batch_size: int = 16,
    grad_accum: int = 1,
    lr: float = 1e-3,
    lora_r: int = 32,
    tts_frac: float = 0.25,
    seed: int = 0,
    val_manifest: str | None = None,
    eval_every: int = 250,
):
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    assert arm in ("clean", "tel", "tel+tts")
    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    proc = WhisperProcessor.from_pretrained(model_id, language="hindi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.config.forced_decoder_ids = None
    model.generation_config.language = "hindi"
    model.generation_config.task = "transcribe"
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, target_modules=LORA_TARGETS))
    model.print_trainable_parameters()
    model.to(device)

    real = read_rows(train_manifest)
    synth = read_rows(tts_manifest) if (arm == "tel+tts" and tts_manifest) else []
    if arm == "tel+tts" and not synth:
        raise ValueError("tel+tts needs --tts-manifest")

    augment = None
    if arm != "clean":
        import soundfile as sf

        noises = [sf.read(f, dtype="float32")[0] for f in (noise_files or [])]
        augment = Augmenter(noises, TelConfig())
    collate = Collator(proc, augment, seed)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    warmup = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    gen = batches(real, synth, batch_size, tts_frac, random.Random(seed))
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        for _ in range(grad_accum):
            feats, labels = collate(next(gen))
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(input_features=feats.to(device), labels=labels.to(device)).loss / grad_accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()

        if step % 25 == 0:
            row = {"step": step, "loss": loss.item() * grad_accum, "lr": sched.get_last_lr()[0], "min": round((time.time() - t0) / 60, 1)}
            log.append(row)
            print(json.dumps(row), flush=True)
        if val_manifest and step % eval_every == 0:
            model.eval()
            w = quick_wer(model, proc, read_rows(val_manifest)[:100], device)
            log.append({"step": step, "val_wer": w})
            print(json.dumps({"step": step, "val_wer": w}), flush=True)
            model.train()

    model.save_pretrained(out / "adapter")
    proc.save_pretrained(out / "adapter")
    (out / "train_log.json").write_text(json.dumps({"arm": arm, "model": model_id, "steps": steps, "batch_size": batch_size * grad_accum, "lr": lr, "lora_r": lora_r, "tts_frac": tts_frac if synth else 0, "seed": seed, "log": log}, indent=1))
    return out / "adapter"


@torch.inference_mode()
def quick_wer(model, proc, rows, device) -> float:
    """Basic-normalised WER on a small validation slice, for the training log
    only. Final numbers come from asrbench."""
    import soundfile as sf
    from asrbench.metrics import corpus_rate, score_all

    hyps = {}
    for i in range(0, len(rows), 16):
        b = rows[i : i + 16]
        feats = proc.feature_extractor([sf.read(r["audio"], dtype="float32")[0] for r in b], sampling_rate=16_000, return_tensors="pt").input_features.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            ids = model.generate(input_features=feats, language="hindi", task="transcribe", max_new_tokens=200)
        for r, t in zip(b, proc.batch_decode(ids, skip_special_tokens=True)):
            hyps[r["id"]] = t
    return corpus_rate(score_all({r["id"]: r["text"] for r in rows}, hyps, "basic"))
