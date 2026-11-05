#!/usr/bin/env bash
# Whole experiment: data, three training arms, evaluation on clean and
# phone-line test sets. Each arm is ~1.5 h on a Kaggle T4 at the defaults.
# Noise comes from the VAD repo's ESC-50 folder (copy or symlink it to data/noise).
set -euo pipefail
STEPS=${STEPS:-2000}
NOISE=${NOISE:-data/noise}

# 1. Real data: FLEURS Hindi train for training, test split for evaluation.
asrbench prepare --dataset google/fleurs --config hi_in --split train --text-field raw_transcription --lang hi --n 100000 --tag train_hi --out data
asrbench prepare --dataset google/fleurs --config hi_in --split validation --text-field raw_transcription --lang hi --n 200 --tag val_hi --out data
asrbench prepare --dataset google/fleurs --config hi_in --split test --text-field raw_transcription --lang hi --n 100000 --tag test_hi --out data
telasr make-test --manifest data/test_hi/manifest.jsonl --noise "$NOISE" --snr 10 --out data/test_hi_tel

# 2. Synthetic speech, round-trip filtered with whisper-large-v3.
python scripts/get_sentences.py --test data/test_hi/manifest.jsonl data/val_hi/manifest.jsonl --n 2000 --out data/tts_sentences.txt
telasr tts --sentences data/tts_sentences.txt --n 1500 --out data/tts
asrbench transcribe whisper --manifest data/tts/manifest_unfiltered.jsonl --model-id openai/whisper-large-v3 --name roundtrip --results data/tts
telasr filter --manifest data/tts/manifest_unfiltered.jsonl --hyps data/tts/hyp_roundtrip.jsonl --max-cer 0.15 --out data/tts/manifest.jsonl

# 3. Train the arms. Same seed, steps and LoRA config for all three.
COMMON="--train-manifest data/train_hi/manifest.jsonl --val-manifest data/val_hi/manifest.jsonl --noise $NOISE --steps $STEPS"
telasr train --arm clean   $COMMON --out results/clean
telasr train --arm tel     $COMMON --out results/tel
telasr train --arm tel+tts $COMMON --tts-manifest data/tts/manifest.jsonl --out results/tel_tts

# 4. Evaluate: zero-shot whisper-small plus the three adapters, on both test sets.
for SET in test_hi test_hi_tel; do
  R=results/eval_$SET
  asrbench transcribe whisper --manifest data/$SET/manifest.jsonl --model-id openai/whisper-small --name small-zeroshot --results $R
  asrbench transcribe whisper --manifest data/$SET/manifest.jsonl --model-id openai/whisper-small --adapter results/clean/adapter   --name small-clean   --results $R
  asrbench transcribe whisper --manifest data/$SET/manifest.jsonl --model-id openai/whisper-small --adapter results/tel/adapter     --name small-tel     --results $R
  asrbench transcribe whisper --manifest data/$SET/manifest.jsonl --model-id openai/whisper-small --adapter results/tel_tts/adapter --name small-tel-tts --results $R
  asrbench report --manifest data/$SET/manifest.jsonl --results $R
  telasr compare --manifest data/$SET/manifest.jsonl --results $R --baseline small-clean --out $R/compare.md
done
