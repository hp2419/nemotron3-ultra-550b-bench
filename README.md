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

## Recommended configuration

`--max-num-seqs` should be chosen per traffic profile:

| traffic profile | `--max-num-seqs` | rationale |
|---|---|---|
| short / interactive (< 2k ISL) | 320 | +28% throughput, 10x better TTFT |
| long context (> 10k ISL) | 128 | 320 provides no benefit and degrades at high concurrency |

MTP `k=2` helps on both shapes and is recommended unconditionally, subject to the
acceptance-rate caveat below.

Full serve commands: [scripts/](scripts/).

## Contents

| path | contents |
|---|---|
| `results/short-shape.md` | 512 in / 256 out, concurrency 8-300 |
| `results/long-shape.md` | 14,000 in / 5,000 out, concurrency 8-300 |
| `results/analysis.md` | Queueing and roofline math behind the findings |
| `scripts/` | Serve commands per variant, benchmark driver, pod manifests |
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
3. One data point (512/256 at c=200 with MTP+320) is non-monotonic against its neighbours and
   is flagged in `results/short-shape.md` pending a rerun.
4. Single node, single run per configuration. No run-to-run variance measured.

## Status

Complete: baseline, MTP k=2, max-num-seqs 320, and the combination, on both endpoint shapes.

Not yet done: ServeIt Studio comparison run, four intermediate workload shapes,
real-text acceptance-rate measurement, nsys profile.
