# whisper-hindi-telephony

Does fine-tuning Whisper on phone-degraded audio help it on phone calls, and does adding synthetic TTS speech help further? Three LoRA runs on whisper-small that differ only in their training data, scored on clean and phone-line Hindi.

Headline result: _ (fill in from results/eval_*/compare.md after the run)

## Setup

All three arms share the model (whisper-small), LoRA config (rank 32 on attention and MLP projections), learning rate, steps, batch size and seed. Only the data changes.

| arm | training data |
|---|---|
| clean | FLEURS Hindi train, as recorded |
| tel | the same audio, degraded on the fly: random gain, noise at 0 to 20 dB SNR (70% of examples), then a phone channel (300 to 3400 Hz, 8 kHz, AMR-NB or mu-law) |
| tel+tts | tel, with 25% of each batch replaced by synthetic speech from Indic Parler-TTS |

The synthetic speech is generated from Hindi Wikipedia sentences, after removing any sentence that shares a 4-gram with the test or validation references. Each clip gets a random speaker and style description. Every clip is then transcribed back with whisper-large-v3 and dropped if its character error rate against the input text is above 15%. TTS models skip and slur words, and a clip whose audio doesn't match its label teaches the wrong thing.

Round-trip filter: _ of _ clips kept, median CER _.

The phone-line test set is a fixed copy of FLEURS Hindi test made once with held-out noise files at 10 dB SNR, so every model is scored on identical audio. It keeps the clean set's utterance ids, which makes clean-vs-phone comparisons paired.

Scoring uses [indic-asr-bench](https://github.com/Xzaos/indic-asr-bench), with indic normalisation and translit-aware matching, bootstrap intervals and paired tests.

## Results

WER (%, indic+translit), 95% bootstrap interval in brackets. Δ is against the clean arm, paired over utterances.

| system | clean test | phone test | Δ vs clean arm (phone) | p |
|---|---|---|---|---|
| whisper-small, zero-shot | _ | _ | _ | _ |
| clean | _ | _ | baseline | |
| tel | _ | _ | _ | _ |
| tel+tts | _ | _ | _ | _ |

Code-switched slice (utterances where the reference or some system writes a word in Latin): _ utterances.

| system | WER code-switched | WER rest |
|---|---|---|
| clean | _ | _ |
| tel | _ | _ |
| tel+tts | _ | _ |

What I read from these: _

If the synthetic arm doesn't beat tel, that's the result and it stays in the table.

## Running it

```bash
pip install -e ".[tts,dev]"
pytest -q
ln -s ../vad-from-scratch/data/noise data/noise   # ESC-50 noise from the VAD repo
bash scripts/run_arms.sh
```

Each arm takes about 1.5 hours on a Kaggle T4 at the default 2000 steps. Everything after training (transcription, scoring) resumes if interrupted.

## Code-switching

FLEURS references write English loanwords in Devanagari, so you can't find code-switched utterances by looking for Latin in the reference alone. `analysis.py` also flags an utterance when any system's hypothesis writes a word in Latin that cross-script matches a reference word, which means at least one model heard it as English. The slice is fixed once from all hypotheses and reused for every arm.

## Limitations

FLEURS is read speech, and the phone test is simulated. A real call-centre set would be the honest test, and the absolute numbers will be worse there.

One seed per arm. A difference inside the bootstrap interval of the paired test shouldn't be read as a real effect.

The round-trip filter uses whisper-large-v3, so the synthetic set is biased toward speech that Whisper already finds easy. A filter using a different model family (IndicConformer) would remove that bias.
