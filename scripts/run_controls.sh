#!/usr/bin/env bash
# Controls 1b (reverse cue) and 3 (ask-the-model), all three sizes.
#
#   1b  reverse / revnull : all three classes -- these are mechanistic
#       reads of the same forward passes and do not address the model
#       in natural language, so the stimulus-sense objection does not
#       bite. Target set = entities (cue is a value).
#   3   ask                : "high" class ONLY. The ask arm addresses
#       the model in English, so it needs a base prompt that is a
#       well-formed sentence; mid/low are not.
#
# fp32 at 0.6B/1.7B, bf16 at 4B (18 GB RAM). run_grid skips any cell
# whose npz already exists, so this is resumable.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
S="0 1 2 3 4 5 6 7 8 9"

for spec in "Qwen/Qwen3-0.6B-Base float32" \
            "Qwen/Qwen3-1.7B-Base float32" \
            "Qwen/Qwen3-4B-Base bfloat16"; do
  set -- $spec
  echo "=== $1 ($2) reverse+revnull, all rho  $(date +%T) ==="
  $PY scripts/run_grid.py --model "$1" --dtype "$2" --seeds $S \
      --rhos low mid high --cues reverse revnull
  echo "=== $1 ($2) ask, high only  $(date +%T) ==="
  $PY scripts/run_grid.py --model "$1" --dtype "$2" --seeds $S \
      --rhos high --cues ask
done
echo "=== ALL DONE $(date +%T) ==="
