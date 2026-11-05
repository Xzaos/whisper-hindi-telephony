"""Synthetic Hindi speech from Indic Parler-TTS, filtered by round-trip ASR.

Text comes from a file of Hindi sentences that do not appear in any test
set. Each sentence gets a random speaker and style description, so the
synthetic set isn't one voice reading everything in the same tone.

TTS sometimes skips words or mumbles. Every clip is transcribed back with a
strong ASR model and dropped if its CER against the input text is above
`max_cer`. Without that filter the model learns from audio that doesn't
match its label, which is worse than not having the clip at all.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

SPEAKERS = ["Rohit", "Divya", "Aman", "Rani", "Sumit", "Kavya", "Ananya", "Karan"]
STYLES = [
    "speaks at a moderate pace with a clear, close-sounding recording",
    "speaks quickly in a slightly expressive tone; the recording is a bit distant",
    "speaks slowly and calmly with a very clear recording",
    "speaks with a high pitch at a normal pace in a quiet room",
    "speaks in a low voice with a little background noise",
]


def describe(rng: random.Random) -> str:
    return f"{rng.choice(SPEAKERS)} {rng.choice(STYLES)}."


def generate(sentences: list[str], out_dir: str, n: int, seed: int = 0, model_id: str = "ai4bharat/indic-parler-tts", device: str | None = None) -> Path:
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
    tok = AutoTokenizer.from_pretrained(model_id)
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr = model.config.sampling_rate

    rng = random.Random(seed)
    picks = rng.sample(sentences, min(n, len(sentences)))
    out = Path(out_dir)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, text in enumerate(picks):
        desc = describe(rng)
        d = desc_tok(desc, return_tensors="pt").to(device)
        p = tok(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            audio = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask, prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
        x = audio.cpu().float().squeeze()
        if sr != 16_000:
            x = AF.resample(x, sr, 16_000)
        path = out / "wav" / f"tts_{i:05d}.wav"
        sf.write(path, x.numpy(), 16_000)
        rows.append({"id": f"tts_{i:05d}", "audio": str(path), "text": text, "lang": "hi", "desc": desc})
    raw = out / "manifest_unfiltered.jsonl"
    raw.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return raw


def roundtrip_filter(manifest: str, hyps: str, out: str, max_cer: float = 0.15) -> dict:
    """hyps is an asrbench hyp file produced by transcribing the unfiltered
    manifest. Keeps clips whose round-trip CER is at most max_cer."""
    from asrbench.data import read_hyps, read_manifest
    from asrbench.metrics import score_utterance

    rows, h = read_manifest(manifest), read_hyps(hyps)
    kept, cers = [], []
    for r in rows:
        s = score_utterance(r["id"], r["text"], h.get(r["id"], ""), "indic")
        cer = s.char_errors / max(1, s.ref_chars)
        cers.append(cer)
        if cer <= max_cer:
            kept.append(r)
    Path(out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    stats = {"total": len(rows), "kept": len(kept), "median_cer": float(np.median(cers)) if cers else None}
    print(stats)
    return stats
