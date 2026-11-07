# Results

Manifest: `data/test_hi_tel/manifest.jsonl`, 417 utterances.

## hi (417 utterances, 10185 reference words)

| system | WER raw | WER basic | WER indic | WER indic+translit | CER indic |
|---|---|---|---|---|---|
| small-clean | 41.87 [39.84, 43.97] | 36.97 [34.93, 39.13] | 36.26 [34.24, 38.43] | 36.19 [34.16, 38.37] | 19.73 [17.98, 21.59] |
| small-tel | 36.09 [34.55, 37.67] | 30.94 [29.33, 32.52] | 30.19 [28.60, 31.80] | 30.12 [28.52, 31.72] | 15.09 [13.92, 16.38] |
| small-zeroshot | 81.63 [77.15, 86.72] | 77.84 [73.26, 82.99] | 77.26 [72.71, 82.44] | 77.23 [72.69, 82.41] | 47.16 [43.54, 51.03] |

WER in %, 95% bootstrap interval over utterances in brackets.

Paired bootstrap vs `small-tel` (indic+translit WER):

| system | ΔWER | p |
|---|---|---|
| small-clean | 6.07 [4.68, 7.51] | 0.000 |
| small-zeroshot | 47.11 [42.60, 52.17] | 0.000 |

Error buckets (basic normalisation, word ops):

| system | orthographic | script_mix | number_format | split_merge | near_miss | lexical | deletion | insertion |
|---|---|---|---|---|---|---|---|---|
| small-clean | 69 | 7 | 63 | 58 | 1096 | 1659 | 509 | 330 |
| small-tel | 76 | 7 | 54 | 100 | 948 | 1377 | 354 | 257 |
| small-zeroshot | 50 | 1 | 89 | 127 | 1350 | 4017 | 937 | 1411 |
