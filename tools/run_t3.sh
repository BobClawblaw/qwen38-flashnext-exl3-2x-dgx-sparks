#!/usr/bin/env bash
# T3 long-context needles against the live serve. TensorFold has no /tokenize, so the needle documents
# are sized with the checkpoint's own tokenizer.json inside the recipe image (QLIB_TOKENIZER).
#   tools/run_t3.sh OUT_DIR [extra t3_needles.py args]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(mkdir -p "$1" && cd "$1" && pwd)"; shift
IMAGE="${IMAGE:-tf-qwen38-flashnext-exl3:0.6.6}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
SNAP=/cache/huggingface/hub/models--turboderp--Qwen3.8-Flash-Next-exl3/snapshots/${SNAPSHOT_SHA:-55a732e0c4c3d4614bc42b68493bb930d9b02c0a}
docker run --rm --network host --user "$(id -u):$(id -g)" \
  -v "$ROOT:/recipe:ro" -v "$OUT:/out" -v "$HF_CACHE:/cache/huggingface:ro" \
  -e QLIB_TOKENIZER="$SNAP/tokenizer.json" -e HOME=/tmp \
  --entrypoint python "$IMAGE" /recipe/quality/t3_needles.py --url "http://127.0.0.1:${PORT:-8000}" --out /out "$@"
