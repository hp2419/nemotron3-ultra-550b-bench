#!/usr/bin/env bash
# Config B - baseline + MTP speculative decoding, k=2.
# --mamba-cache-mode align is required: MTP combined with chunked prefill has a known
# accuracy issue on Mamba layers without it.
set -euo pipefail
MODEL=${MODEL:-RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-dynamic}
LOG=${LOG:-/workspace/logs/serve_mtp2.log}

vllm serve "$MODEL" \
  --port 8000 \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --max-num-seqs 128 \
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
