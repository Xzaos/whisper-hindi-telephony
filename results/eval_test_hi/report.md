# Results

Manifest: `data/test_hi/manifest.jsonl`, 417 utterances.

## hi (417 utterances, 10185 reference words)

| system | WER raw | WER basic | WER indic | WER indic+translit | CER indic |
|---|---|---|---|---|---|
| small-clean | 32.38 [30.72, 34.00] | 27.16 [25.50, 28.79] | 26.32 [24.61, 27.99] | 26.23 [24.53, 27.90] | 12.45 [11.16, 13.91] |
| small-tel | 30.84 [29.41, 32.35] | 25.57 [24.17, 27.07] | 24.70 [23.30, 26.21] | 24.60 [23.19, 26.14] | 11.32 [10.22, 12.54] |
| small-zeroshot | 70.81 [66.86, 75.21] | 65.98 [61.96, 70.42] | 65.05 [61.05, 69.55] | 64.96 [60.98, 69.50] | 35.83 [32.81, 39.21] |

WER in %, 95% bootstrap interval over utterances in brackets.

Paired bootstrap vs `small-tel` (indic+translit WER):

| system | ΔWER | p |
|---|---|---|
| small-clean | 1.63 [0.27, 2.97] | 0.020 |
| small-zeroshot | 40.36 [36.31, 44.75] | 0.000 |

Error buckets (basic normalisation, word ops):

| system | orthographic | script_mix | number_format | split_merge | near_miss | lexical | deletion | insertion |
|---|---|---|---|---|---|---|---|---|
| small-clean | 84 | 9 | 45 | 104 | 1003 | 923 | 397 | 220 |
| small-tel | 87 | 10 | 50 | 130 | 887 | 926 | 315 | 217 |
| small-zeroshot | 90 | 6 | 82 | 160 | 1597 | 3098 | 682 | 1051 |
