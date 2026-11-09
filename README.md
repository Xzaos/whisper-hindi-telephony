# Whisper Hindi Telephony

Does fine-tuning Whisper on phone-degraded audio help it on phone calls, and does adding synthetic TTS speech help further? I trained two LoRA arms on whisper-small that differ only in their training data and scored them on clean and phone-line Hindi. The third arm, with synthetic speech, did not run (see Results).

On the phone test set, the tel arm beats the clean arm by 6.07 WER points (95% CI 4.68 to 7.51, p<0.001). On the clean test set it is also better, by 1.63 (0.27 to 2.97, p=0.020).

## Setup

Both arms share the model (whisper-small), LoRA config (rank 32 on attention and MLP projections), learning rate, steps (1500), batch size (8) and seed (0). Batch size 16 ran out of memory on a T4. Only the data changes.

| arm | training data |
|---|---|
| clean | FLEURS Hindi train, as recorded |
| tel | the same audio, degraded on the fly: random gain, noise at 0 to 20 dB SNR (70% of examples), then a phone channel (300 to 3400 Hz, 8 kHz, AMR-NB or mu-law) |
| tel+tts | tel, with 25% of each batch replaced by synthetic speech from Indic Parler-TTS (did not run) |

The tel+tts code is in the repo and tested: sentences come from Hindi Wikipedia with any 4-gram overlap with the test or validation references removed, each clip gets a random speaker and style description, and clips are transcribed back with whisper-large-v3 and dropped if the character error rate against the input text is above 15%. Indic Parler-TTS generation stalled on Kaggle and hit the 12-hour limit before finishing 500 clips, so there is no filtered set and no result for this arm.

The phone-line test set is a fixed copy of FLEURS Hindi test made once with held-out noise files at 10 dB SNR, so every model is scored on identical audio. It keeps the clean set's utterance ids, which makes clean-vs-phone comparisons paired.

Scoring uses [indic-asr-bench](https://github.com/Xzaos/indic-asr-bench), with indic normalisation and translit-aware matching, bootstrap intervals and paired tests. Both test sets have 417 utterances and 10185 reference words.

## Results

WER (%, indic+translit), 95% bootstrap interval in brackets. Δ is against the clean arm on the same test set, paired over utterances. Raw output is in `results/eval_test_hi` and `results/eval_test_hi_tel`.

| system | clean test | Δ (clean test) | phone test | Δ (phone test) |
|---|---|---|---|---|
| whisper-small, zero-shot | 64.96 [60.98, 69.50] | +38.73 [34.81, 42.95] | 77.23 [72.69, 82.41] | +41.04 [36.35, 46.10] |
| clean | 26.23 [24.53, 27.90] | baseline | 36.19 [34.16, 38.37] | baseline |
| tel | 24.60 [23.19, 26.14] | -1.63 [-2.97, -0.27], p=0.020 | 30.12 [28.52, 31.72] | -6.07 [-7.51, -4.68], p<0.001 |

Going from the clean to the phone test set costs the clean arm 10.0 points and the tel arm 5.5. Both arms are far better than zero-shot whisper-small.

Word errors on the phone test set, by bucket (basic normalisation):

| system | lexical | deletion | insertion | near miss | split/merge |
|---|---|---|---|---|---|
| clean | 1659 | 509 | 330 | 1096 | 58 |
| tel | 1377 | 354 | 257 | 948 | 100 |

Most of the gain comes from fewer deletions (509 to 354) and fewer lexical errors (1659 to 1377). The tel arm makes more split and merge errors (58 to 100). The other buckets are in the report files.

Code-switched slice (utterances where the reference or some system writes a word in Latin): 17 utterances on the clean test set, 16 on the phone test set.

| system | code-switched, clean test | code-switched, phone test |
|---|---|---|
| clean | 31.14 [21.70, 41.59] | 43.11 [27.57, 66.85] |
| tel | 29.44 [21.05, 38.56] | 35.20 [26.10, 45.15] |

The slice is too small and the intervals too wide to conclude anything from it. The tel arm has the lower point estimate in both, but the intervals overlap heavily.

The tel+tts arm has no results. I can't say whether synthetic speech helps on top of telephony augmentation.

## Running it

```bash
pip install -e ".[tts,dev]"
pytest -q
ln -s ../vad-from-scratch/data/noise data/noise   # ESC-50 noise from the VAD repo
bash scripts/run_arms.sh
```

The script defaults to 2000 steps and batch size 16. The results here used `STEPS=1500` and batch size 8, which is what fit on a T4. Everything after training (transcription, scoring) resumes if interrupted.

## Code-switching

FLEURS references write English loanwords in Devanagari, so you can't find code-switched utterances by looking for Latin in the reference alone. `analysis.py` also flags an utterance when any system's hypothesis writes a word in Latin that cross-script matches a reference word, which means at least one model heard it as English. The slice is fixed once from all hypotheses and reused for every arm.

## Limitations

FLEURS is read speech, and the phone test is simulated. A real call-centre set would be the honest test, and the absolute numbers will be worse there.

One seed per arm. A difference inside the bootstrap interval of the paired test shouldn't be read as a real effect.

The synthetic arm did not run, so the question of whether TTS speech adds anything is open. If it is rerun, the round-trip filter uses whisper-large-v3, which biases the synthetic set toward speech that Whisper already finds easy. A filter using a different model family (IndicConformer) would remove that bias.
