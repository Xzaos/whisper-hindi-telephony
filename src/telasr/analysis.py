"""Per-arm comparison on the clean and phone-line test sets, plus the
code-switching slice.

Code-switched utterances are found two ways, since FLEURS-style Hindi
references write English loanwords in Devanagari:
  1. the reference contains a Latin token, or
  2. some system's hypothesis writes a word in Latin that cross-script
     matches a reference word (i.e. a model heard it as English).
The second rule depends on the systems being compared, so the slice is
fixed once from all hypotheses and then reused for every arm.
"""
from __future__ import annotations

from pathlib import Path

from asrbench.align import align
from asrbench.data import read_hyps, read_manifest
from asrbench.metrics import corpus_rate, score_all
from asrbench.normalize import is_latin_token, normalize
from asrbench.stats import bootstrap_ci, paired_bootstrap
from asrbench.translit import cross_script_match


def code_switch_ids(refs: dict[str, str], hyp_files: list[Path]) -> set[str]:
    ids = {u for u, t in refs.items() if any(is_latin_token(w) for w in t.split())}
    for f in hyp_files:
        for u, h in read_hyps(f).items():
            if u not in refs:
                continue
            r, hh = normalize(refs[u], "indic").split(), normalize(h, "indic").split()
            for op in align(r, hh, cross_script_match).ops:
                if op.kind == "eq" and op.ref != op.hyp:
                    ids.add(u)
                    break
    return ids


def compare(manifest: str, results_dir: str, baseline: str, mode: str = "indic+translit") -> str:
    refs = {r["id"]: r["text"] for r in read_manifest(manifest)}
    files = sorted(Path(results_dir).glob("hyp_*.jsonl"))
    cs = code_switch_ids(refs, files)
    lines = [f"{len(refs)} utterances, {len(cs)} code-switched. Scoring: {mode}.\n", "| system | WER all | WER code-switched | WER rest | Δ vs baseline (all) | p |", "|---|---|---|---|---|---|"]
    scored = {f.stem.removeprefix("hyp_"): score_all(refs, {k: v for k, v in read_hyps(f).items() if k in refs}, mode) for f in files}
    if baseline not in scored:
        raise SystemExit(f"baseline {baseline} not in {list(scored)}")
    base = scored[baseline]
    for name, sc in scored.items():
        sub = lambda keep: [s for s in sc if keep(s.id)]
        ci = lambda xs: bootstrap_ci([s.word_errors for s in xs], [s.ref_words for s in xs]).fmt() if xs else "n/a"
        if name == baseline:
            delta, p = "baseline", ""
        else:
            res = paired_bootstrap([s.word_errors for s in sc], [s.word_errors for s in base], [s.ref_words for s in base])
            delta, p = res.diff.fmt(), f"{res.p_value:.3f}"
        lines.append(f"| {name} | {ci(sc)} | {ci(sub(lambda u: u in cs))} | {ci(sub(lambda u: u not in cs))} | {delta} | {p} |")
    return "\n".join(lines) + "\n"
