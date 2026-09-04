#!/usr/bin/env bash
# Fourth-arm disambiguation: repeat / repeatphrase, high class, seeds 0-9.
# `relation` collapsed like `cued`, so the ask effect is framing OR cue
# repetition; these hold repetition and drop the interrogative.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
for spec in "Qwen/Qwen3-0.6B-Base float32" "Qwen/Qwen3-1.7B-Base float32" "Qwen/Qwen3-4B-Base bfloat16"; do
  set -- $spec
  echo "=== $1 ($2)  $(date +%T) ==="
  $PY scripts/run_grid.py --model "$1" --dtype "$2" \
      --seeds 0 1 2 3 4 5 6 7 8 9 --rhos high --cues repeat repeatphrase
done
echo "=== ARMS3 DONE $(date +%T) ==="
