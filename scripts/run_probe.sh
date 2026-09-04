#!/usr/bin/env bash
# Control 2 capture: residual stream at answer_pos, seeds 0-29,
# all N, all rho, cued arm only. fp32 at 0.6B/1.7B, bf16 at 4B.
# Resumable at model granularity (skips a model whose npz exists).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}

for spec in "Qwen/Qwen3-0.6B-Base float32" \
            "Qwen/Qwen3-1.7B-Base float32" \
            "Qwen/Qwen3-4B-Base bfloat16"; do
  set -- $spec
  echo "=== $1 ($2)  $(date +%T) ==="
  $PY scripts/capture_resid.py --model "$1" --dtype "$2"
done
echo "=== PROBE CAPTURE DONE $(date +%T) ==="
