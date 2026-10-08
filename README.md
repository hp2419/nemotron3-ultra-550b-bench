# Nemotron-3-Ultra-550B-A55B-FP8 — vLLM tuning on 1x 8xH200

Interactive-serving tuning study for `RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-dynamic`
on a single 8xH200 node (TP8, expert parallel), targeting low TTFT and low inter-token latency.

Engine: upstream `vllm/vllm-openai:latest` (vLLM 0.29.0, torch 2.13.0+cu130).
Benchmark: guidellm, synthetic text.

## Headline result

`--max-num-seqs` was the binding constraint at high concurrency, not the hardware.
Raising it from 128 to 320 plus MTP speculative decoding (k=2) gives, on 512/256 at 300 concurrent:

| metric | stock config | tuned | change |
|---|---|---|---|
| TTFT | 14,964 ms | 1,493 ms | **-90%** |
| Output throughput | ~2,593 tok/s | 3,328 tok/s | **+28%** |
| E2E latency | ~26 s | 22.3 s | -14% |
| ITL | ~44 ms | 81.6 ms | +85% |

The ITL regression is the cost of a larger batch. Total time is still lower and first-token
latency is 10x better, which is the right trade for interactive serving.

**The same change does not help on long-context traffic** (14k in / 5k out) and hurts above
128 concurrent. See [results/long-shape.md](results/long-shape.md).

## Configurations tested

Four configurations, differing only in the rows marked below. Everything else is identical
across all four.

| | A: baseline | B: MTP | C: 320 seqs | D: tuned |
|---|---|---|---|---|
| `--max-num-seqs` | 128 | 128 | **320** | **320** |
| `--mamba-cache-mode align` | - | **yes** | - | **yes** |
| `--speculative-config` mtp k | - | **2** | - | **2** |
| script | [serve-baseline.sh](scripts/serve-baseline.sh) | [serve-mtp2.sh](scripts/serve-mtp2.sh) | [serve-seqs320.sh](scripts/serve-seqs320.sh) | [serve-tuned.sh](scripts/serve-tuned.sh) |

Config A reproduces the stock deployment configuration and is the reference for all deltas.

### Common flags (all four configurations)

```
--tensor-parallel-size 8
--trust-remote-code
--max-model-len 64000
--max-num-batched-tokens 8192
--gpu-memory-utilization 0.9
--disable-cascade-attn
--enable-expert-parallel
--enable-chunked-prefill
--enable-prefix-caching
--reasoning-parser nemotron_v3
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
--mamba-ssm-cache-dtype float16
--mamba-backend flashinfer
--enable-mamba-cache-stochastic-rounding
--mamba-cache-philox-rounds 5
--kv-cache-dtype fp8
```

### Per-configuration additions

**A — baseline (stock)**
```
--max-num-seqs 128
```

**B — MTP k=2**
```
--max-num-seqs 128
--mamba-cache-mode align
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

**C — max-num-seqs 320**
```
--max-num-seqs 320
```

**D — tuned (recommended for short / interactive traffic)**
```
--max-num-seqs 320
--mamba-cache-mode align
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

### Notes on specific flags

- **`--mamba-cache-mode align` is mandatory with MTP.** MTP combined with chunked prefill has a
  known accuracy issue on Mamba layers without it. It is not an optimisation; it is a
  correctness requirement, and it is the reason configs B and D carry it while A and C do not.
- **`--speculative-config` method name.** This build accepts `"mtp"`. The RedHatAI model card
  uses `"nemotron_h_mtp"` with k=5. Which name is valid depends on the vLLM version; verify with
  `vllm serve --help=all` before committing GPU time, since vLLM exits on unknown arguments.
- **`num_speculative_tokens`.** k=5 (as on the model card) is too high for this workload.
  Guidance received was k=1 for throughput and k=2 for latency. Only k=2 was measured here.
- **`--gpu-memory-utilization 0.9`** is vLLM's default and is pinned explicitly so that any
  orchestration layer that sets it cannot change the comparison.
- **`--block-size`** is deliberately left unset; the Mamba-hybrid path aligns it itself.
- **`--served-model-name`** is deliberately unset, because guidellm sends the HF model ID.
- **Request logging** (`--enable-request-id-headers`, `--enable-log-requests`, `--max-log-len 0`)
  was present in the stock deployment but is omitted from all runs here, so it does not
  contribute to any published number. Its cost was not measured separately.

### Benchmark parameters

| shape | ISL | OSL | concurrency levels | duration per level |
|---|---|---|---|---|
| short | 512 (528 actual) | 256 | 8, 16, 32, 64, 128, 200, 300 | 300 s |
| long | 14,000 (14,016 actual) | 5,000 | 8, 32, 128, 300 | 900 s |

"Actual" is what guidellm's synthetic generator emits after rounding to a tokenizer boundary.
Driver: [scripts/bench.sh](scripts/bench.sh).

## Recommended configuration

`--max-num-seqs` should be chosen per traffic profile:

| traffic profile | `--max-num-seqs` | rationale |
|---|---|---|
| short / interactive (< 2k ISL) | 320 | +28% throughput, 10x better TTFT |
| long context (> 10k ISL) | 128 | 320 provides no benefit and degrades at high concurrency |

MTP `k=2` helps on both shapes and is recommended unconditionally, subject to the
acceptance-rate caveat below.

## Contents

| path | contents |
|---|---|
| `results/short-shape.md` | 512 in / 256 out, concurrency 8-300 |
| `results/long-shape.md` | 14,000 in / 5,000 out, concurrency 8-300 |
| `results/analysis.md` | Queueing and roofline math behind the findings |
| `results/raw/` | Raw guidellm JSON per run |
| `scripts/` | Serve commands per variant, benchmark driver, pod manifest |
| `docs/methodology.md` | Environment, procedure, how to reproduce |
| `docs/open-questions.md` | Untested knobs and known gaps |

## Caveats

1. **All measurements use guidellm `synthetic_text` (randomly sampled tokens).** Latency and
   throughput figures are content-independent and stand. The MTP draft-acceptance rate is
   **not** content-independent: 68% on the short shape and 97% on the long shape. The 97%
   figure is an artifact of low-entropy random-token continuations and should not be quoted.
   A rerun on representative text is required before publishing any acceptance figure.
2. Acceptance was derived from `output_tokens / stream_iterations`, not from vLLM's
   `spec_decode` counters. Same ballpark, but not the instrumented number.
3. One data point (512/256 at c=200 with config D) is non-monotonic against its neighbours and
   is flagged in `results/short-shape.md` pending a rerun.

