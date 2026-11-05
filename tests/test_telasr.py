import json

import numpy as np

from telasr.analysis import code_switch_ids, compare
from telasr.augment import SR, Augmenter, TelConfig, add_noise, telephony


def tone(sec=1.0, f=1000.0):
    t = np.arange(int(sec * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def band_energy(x, lo, hi):
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    return spec[(freqs >= lo) & (freqs < hi)].sum()


def test_telephony_removes_high_band():
    rng = np.random.default_rng(0)
    x = np.random.default_rng(1).normal(0, 0.2, SR).astype(np.float32)
    y = telephony(x, rng, p_amr=0.0)
    assert y.shape == x.shape
    # almost nothing left above 4 kHz after the 8 kHz round trip
    assert band_energy(y, 4500, 8000) < 0.01 * band_energy(x, 4500, 8000)
    assert band_energy(y, 500, 3000) > 0.2 * band_energy(x, 500, 3000)


def test_add_noise_hits_target_snr():
    rng = np.random.default_rng(0)
    x = tone()
    n = np.random.default_rng(2).normal(0, 1, SR // 3).astype(np.float32)
    y = add_noise(x, n, 5.0, rng)
    snr = 10 * np.log10(np.mean(x**2) / np.mean((y - x) ** 2))
    assert abs(snr - 5.0) < 0.1


def test_augmenter_output_is_bounded():
    aug = Augmenter([np.ones(SR, np.float32)], TelConfig(p_amr=0.0))
    y = aug(tone(), np.random.default_rng(0))
    assert np.abs(y).max() <= 1.0 and y.dtype == np.float32


def _write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_code_switch_slice_and_compare(tmp_path):
    refs = [
        {"id": "a", "text": "मेरा मोबाइल खो गया", "lang": "hi", "audio": ""},
        {"id": "b", "text": "आज बारिश हुई", "lang": "hi", "audio": ""},
        {"id": "c", "text": "मुझे laptop चाहिए", "lang": "hi", "audio": ""},
    ]
    _write(tmp_path / "m.jsonl", refs)
    res = tmp_path / "res"
    res.mkdir()
    _write(res / "hyp_base.jsonl", [{"id": "a", "hyp": "मेरा mobile खो गया"}, {"id": "b", "hyp": "आज बारिश"}, {"id": "c", "hyp": "मुझे चाहिए"}])
    _write(res / "hyp_tel.jsonl", [{"id": "a", "hyp": "मेरा मोबाइल खो गया"}, {"id": "b", "hyp": "आज बारिश हुई"}, {"id": "c", "hyp": "मुझे laptop चाहिए"}])
    ids = code_switch_ids({r["id"]: r["text"] for r in refs}, sorted(res.glob("hyp_*.jsonl")))
    assert ids == {"a", "c"}
    table = compare(str(tmp_path / "m.jsonl"), str(res), "base")
    assert "| tel |" in table and "2 code-switched" in table
