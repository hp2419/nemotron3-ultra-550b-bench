#!/usr/bin/env bash
# Config C - baseline with --max-num-seqs raised to 320, no speculative decoding.
set -euo pipefail
MODEL=${MODEL:-RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-dynamic}
LOG=${LOG:-/workspace/logs/serve_seqs320.log}

vllm serve "$MODEL" \
  --port 8000 \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --max-num-seqs 320 \
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
  2>&1 | tee "$LOG"
