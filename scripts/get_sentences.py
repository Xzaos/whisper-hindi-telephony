"""Pull Hindi sentences for TTS from Hindi Wikipedia, excluding anything that
overlaps the test set.

    python scripts/get_sentences.py --test data/fleurs_hi/manifest.jsonl --n 3000 --out data/tts_sentences.txt
"""
import argparse
import json
import re

from asrbench.normalize import normalize

p = argparse.ArgumentParser()
p.add_argument("--test", required=True, nargs="+")
p.add_argument("--n", type=int, default=3000)
p.add_argument("--out", required=True)
a = p.parse_args()

# Exclude by 4-gram overlap as well as exact match, so a near-copy of a test
# sentence can't leak in.
test_grams = set()
for path in a.test:
    for line in open(path, encoding="utf-8"):
        w = normalize(json.loads(line)["text"], "indic").split()
        test_grams.update(tuple(w[i : i + 4]) for i in range(len(w) - 3))

from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train", streaming=True)
split = re.compile(r"(?<=[।?!])\s+")
devanagari_only = re.compile(r"^[ऀ-ॿ\s,]+[।?!]?$")
out, seen = [], set()
for art in ds:
    for s in split.split(art["text"].replace("\n", " ")):
        s = s.strip()
        w = s.split()
        if not (6 <= len(w) <= 20) or not devanagari_only.match(s):
            continue
        nw = normalize(s, "indic").split()
        if any(tuple(nw[i : i + 4]) in test_grams for i in range(len(nw) - 3)) or s in seen:
            continue
        seen.add(s)
        out.append(s)
    if len(out) >= a.n:
        break

open(a.out, "w", encoding="utf-8").write("\n".join(out[: a.n]) + "\n")
print(f"wrote {min(len(out), a.n)} sentences to {a.out}")
