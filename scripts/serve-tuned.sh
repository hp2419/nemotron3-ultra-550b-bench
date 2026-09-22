#!/usr/bin/env bash
# Config D - recommended for short / interactive traffic.
# --max-num-seqs 320 + MTP k=2. For long-context traffic (>10k ISL) use 128 instead;
# see results/long-shape.md.
set -euo pipefail
MODEL=${MODEL:-RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-dynamic}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-320}
LOG=${LOG:-/workspace/logs/serve_tuned.log}

vllm serve "$MODEL" \
  --port 8000 \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-model-len 64000 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --disable-cascade-attn \
  --enable-expert-parallel \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --mamba-ssm-cache-dtype float16 \
  --mamba-backend flashinfer \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --kv-cache-dtype fp8 \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  2>&1 | tee "$LOG"
