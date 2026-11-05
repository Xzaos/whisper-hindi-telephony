"""Phone-line degradation for training and test audio.

    band-limit 300-3400 Hz -> resample to 8 kHz -> mu-law (or AMR-NB via
    ffmpeg if available) -> back to 16 kHz -> optional additive noise

Training applies it on the fly with random settings. Test sets are made once
with `make_test_set`, from held-out noise files and a fixed seed, so every
model is scored on byte-identical audio.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF

SR = 16_000


@dataclass
class TelConfig:
    p_telephony: float = 1.0
    p_noise: float = 0.7
    snr_db: tuple[float, float] = (0.0, 20.0)
    p_amr: float = 0.5  # only used when ffmpeg has libopencore_amrnb
    gain_db: tuple[float, float] = (-6.0, 3.0)


def _has_amr() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return "amr_nb" in out or "libopencore_amrnb" in out


_AMR = None


def amr_roundtrip(x8k: np.ndarray, bitrate: str = "12.2k") -> np.ndarray:
    import soundfile as sf

    with tempfile.TemporaryDirectory() as d:
        src, enc, dec = Path(d, "a.wav"), Path(d, "a.amr"), Path(d, "b.wav")
        sf.write(src, x8k, 8000)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ar", "8000", "-ac", "1", "-c:a", "libopencore_amrnb", "-b:a", bitrate, enc], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", enc, "-ar", "8000", dec], check=True)
        y, _ = sf.read(dec, dtype="float32")
    return y[: len(x8k)] if len(y) >= len(x8k) else np.pad(y, (0, len(x8k) - len(y)))


def telephony(x: np.ndarray, rng: np.random.Generator, p_amr: float = 0.5) -> np.ndarray:
    global _AMR
    if _AMR is None:
        _AMR = _has_amr()
    t = torch.from_numpy(x.astype(np.float32))
    t = AF.lowpass_biquad(AF.highpass_biquad(t, SR, 300.0), SR, 3400.0)
    t8 = AF.resample(t, SR, 8000).clamp(-1, 1)
    if _AMR and rng.random() < p_amr:
        t8 = torch.from_numpy(amr_roundtrip(t8.numpy()))
    else:
        t8 = AF.mu_law_decoding(AF.mu_law_encoding(t8, 256), 256)
    y = AF.resample(t8, 8000, SR).numpy()
    return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    start = rng.integers(0, len(noise) - len(x) + 1)
    n = noise[start : start + len(x)]
    px, pn = np.mean(x**2) + 1e-12, np.mean(n**2) + 1e-12
    return x + n * np.sqrt(px / (pn * 10 ** (snr_db / 10)))


class Augmenter:
    def __init__(self, noises: list[np.ndarray], cfg: TelConfig = TelConfig()):
        self.noises, self.cfg = noises, cfg

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        c = self.cfg
        x = x * 10 ** (rng.uniform(*c.gain_db) / 20)
        if self.noises and rng.random() < c.p_noise:
            x = add_noise(x, self.noises[rng.integers(len(self.noises))], rng.uniform(*c.snr_db), rng)
        if rng.random() < c.p_telephony:
            x = telephony(x, rng, c.p_amr)
        peak = np.abs(x).max()
        return (x * (0.99 / peak) if peak > 0.99 else x).astype(np.float32)


def make_test_set(manifest: str, out_dir: str, noises: list[np.ndarray], seed: int = 0, snr_db: float = 10.0) -> Path:
    """Telephony copy of a test manifest. Same ids, so asrbench can pair the
    clean and phone-line scores utterance by utterance."""
    import json

    import soundfile as sf

    rng = np.random.default_rng(seed)
    cfg = TelConfig(p_telephony=1.0, p_noise=1.0 if noises else 0.0, snr_db=(snr_db, snr_db), p_amr=1.0, gain_db=(0, 0))
    aug = Augmenter(noises, cfg)
    out = Path(out_dir)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    rows = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            x, _ = sf.read(r["audio"], dtype="float32")
            y = aug(x, rng)
            p = out / "wav" / Path(r["audio"]).name
            sf.write(p, y, SR)
            rows.append({**r, "audio": str(p)})
    m = out / "manifest.jsonl"
    m.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return m
