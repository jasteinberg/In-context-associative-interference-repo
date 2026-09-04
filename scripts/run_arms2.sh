#!/usr/bin/env bash
# Second wave of cue arms, `high` class only, seeds 0-9:
#   asknull   -- D1b twin for the ask arm (was missing)
#   relation  -- explicit relation phrasing, ONE entity mention, no
#                question. Disambiguates the ask effect: ask differs
#                from plain completion in framing AND cue recency, and
#                this arm removes the framing while holding mentions
#                at 1.
#   relnull   -- its D1b twin
# Waits for the probe capture to finish first (RAM: 18 GB total).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}

while pgrep -f run_probe.sh > /dev/null; do sleep 30; done
echo "=== probe capture clear, starting arms  $(date +%T) ==="

for spec in "Qwen/Qwen3-0.6B-Base float32" \
            "Qwen/Qwen3-1.7B-Base float32" \
            "Qwen/Qwen3-4B-Base bfloat16"; do
  set -- $spec
  echo "=== $1 ($2)  $(date +%T) ==="
  $PY scripts/run_grid.py --model "$1" --dtype "$2" \
      --seeds 0 1 2 3 4 5 6 7 8 9 --rhos high \
      --cues asknull relation relnull
done
echo "=== ARMS2 DONE $(date +%T) ==="
