from __future__ import annotations

import argparse
from pathlib import Path


def _noise_files(root: str | None) -> list[str]:
    if not root:
        return []
    return sorted(str(p) for ext in ("wav", "flac") for p in Path(root).rglob(f"*.{ext}"))


def _split_noise(files: list[str]) -> tuple[list[str], list[str]]:
    # Same hashing idea as the VAD repo: test noise never appears in training.
    import hashlib

    train, test = [], []
    for f in files:
        h = int(hashlib.md5(Path(f).name.encode()).hexdigest(), 16) % 5
        (test if h == 0 else train).append(f)
    return train, test


def cmd_make_test(a):
    import soundfile as sf

    from .augment import make_test_set

    _, test_noise = _split_noise(_noise_files(a.noise))
    noises = [sf.read(f, dtype="float32")[0] for f in test_noise]
    print(make_test_set(a.manifest, a.out, noises, seed=a.seed, snr_db=a.snr))


def cmd_tts(a):
    from .tts_synth import generate

    sents = [l.strip() for l in open(a.sentences, encoding="utf-8") if l.strip()]
    print(generate(sents, a.out, a.n, seed=a.seed))


def cmd_filter(a):
    from .tts_synth import roundtrip_filter

    roundtrip_filter(a.manifest, a.hyps, a.out, a.max_cer)


def cmd_train(a):
    from .train import train

    train_noise, _ = _split_noise(_noise_files(a.noise))
    train(a.arm, a.train_manifest, a.out, tts_manifest=a.tts_manifest, noise_files=train_noise, model_id=a.model_id, steps=a.steps, batch_size=a.batch_size, grad_accum=a.grad_accum, lr=a.lr, lora_r=a.lora_r, tts_frac=a.tts_frac, seed=a.seed, val_manifest=a.val_manifest)


def cmd_compare(a):
    from .analysis import compare

    text = compare(a.manifest, a.results, a.baseline)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(prog="telasr")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("make-test", help="phone-line copy of a test manifest")
    s.add_argument("--manifest", required=True)
    s.add_argument("--noise")
    s.add_argument("--snr", type=float, default=10.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/test_tel")
    s.set_defaults(fn=cmd_make_test)

    s = sub.add_parser("tts", help="generate synthetic Hindi speech")
    s.add_argument("--sentences", required=True)
    s.add_argument("--n", type=int, default=1500)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/tts")
    s.set_defaults(fn=cmd_tts)

    s = sub.add_parser("filter", help="drop TTS clips that fail round-trip ASR")
    s.add_argument("--manifest", required=True)
    s.add_argument("--hyps", required=True)
    s.add_argument("--max-cer", type=float, default=0.15)
    s.add_argument("--out", default="data/tts/manifest.jsonl")
    s.set_defaults(fn=cmd_filter)

    s = sub.add_parser("train")
    s.add_argument("--arm", required=True, choices=["clean", "tel", "tel+tts"])
    s.add_argument("--train-manifest", required=True)
    s.add_argument("--val-manifest")
    s.add_argument("--tts-manifest")
    s.add_argument("--noise")
    s.add_argument("--model-id", default="openai/whisper-small")
    s.add_argument("--steps", type=int, default=2000)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--grad-accum", type=int, default=1)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--lora-r", type=int, default=32)
    s.add_argument("--tts-frac", type=float, default=0.25)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("compare", help="per-arm WER with code-switch slice")
    s.add_argument("--manifest", required=True)
    s.add_argument("--results", required=True)
    s.add_argument("--baseline", required=True)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_compare)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
