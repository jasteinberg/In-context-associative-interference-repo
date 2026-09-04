#!/bin/zsh
# Qwen3 base ladder for the in-context load sweep. Base (not -Instruct) models.
export HF_HUB_ENABLE_HF_TRANSFER=0
PY=${PY:-python}
for m in Qwen/Qwen3-0.6B-Base Qwen/Qwen3-1.7B-Base Qwen/Qwen3-4B-Base Qwen/Qwen3-8B-Base; do
  echo "=== $m $(date)"
  $PY -c "from huggingface_hub import snapshot_download as s; print(s('$m', allow_patterns=['*.json','*.safetensors','*.txt','*.py','tokenizer*','merges*','vocab*']))" || echo "FAILED $m"
done
echo "=== ALL DONE $(date)"
