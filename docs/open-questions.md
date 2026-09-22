# Open questions and untested knobs

## Needs a rerun

- **MTP acceptance on representative text.** Both acceptance figures (68% short, 97% long) come
  from random-token synthetic prompts. ShareGPT or sampled production traffic would give a
  number worth publishing. The 97% figure in particular should not leave this repo.
- **512/256 at c=200 with MTP+320.** Non-monotonic against its neighbours; likely an unlucky
  300 s window.
- **TTFT discrepancy at c=8 on the long shape.** Baseline 3.26 s vs 1.06 s for both MTP runs.
  MTP should not affect prefill. Suspect prefix-cache hit-rate differences between runs, or a
  prefill path change from `--mamba-cache-mode align`.
- **Run-to-run variance.** Every configuration was measured once.

## Untested configuration

| knob | hypothesis | cost to test |
|---|---|---|
| `--max-num-seqs 512` | Diminishing: 128->200 gave +14%, 200->300 gave +5.9%. Probably a few percent for a large ITL penalty. | 1 sweep |
| `--max-num-batched-tokens` 16384 / 32768 | Should improve TTFT on the long shape where every 14k prompt needs 2+ prefill chunks. Trades against ITL for requests already decoding. | 2 sweeps |
| `--enable-expert-parallel` off | On one NVLink node, TP-sharded experts (all-reduce) vs EP (all-to-all) could go either way. | 1 sweep |
| `--moe-backend` variants | deep_gemm / flashinfer_cutlass / triton. Unknown which is default in this build. | 2-3 sweeps |
| `--enable-dbo` (dual batch overlap) | Hides all-to-all behind compute. | 1 sweep |
| `--enable-eplb` (expert load balancing) | With 512 experts, hot-expert imbalance stalls the whole step. | 1 sweep |
| `--async-scheduling` | Hides CPU scheduling behind GPU work; matters more as batch grows. May conflict with spec decode in some vLLM versions. | 1 sweep |
| `--mamba-backend triton` + `--mamba-ssm-cache-dtype float32` | Alternative reference configuration; accuracy/perf trade unknown. | 1 sweep |
| `--long-prefill-token-threshold`, `--prefix-match-unit 16` | Present in an NVIDIA reference command; values not obtained. | - |
| MTP k=1 | Reported to favour throughput where k=2 favours latency. Untested here. | 2 sweeps |

## Structural options beyond flags

- **NVFP4 checkpoint.** Halves weight bytes read per decode step, which directly raises the
  decode ceiling, and would make TP4 feasible (TP4 is infeasible at FP8: ~141 GiB needed
  against 139.8 GiB usable).
- **Additional nodes.** Capacity scales roughly linearly; ~10-13 req/s per node on 512/256.
- **Prefill/decode disaggregation** across 2 nodes. Directly attacks the TTFT-vs-ITL conflict
  visible in every table here, but needs a second node and RDMA.

## Profiling

The roofline estimate in `results/analysis.md` puts the expert-weight read at ~13.5 ms against a
measured ~44 ms step at batch 128. The remaining ~30 ms is unattributed. An nsys profile at
c=128 would split it between expert GEMMs, all-to-all, Mamba, attention and scheduling gaps, and
would turn the MoE amortisation hypothesis into a measurement. This is the highest-value
follow-up if flag tuning is exhausted.

## Not yet done

- **ServeIt Studio comparison run.** Requires the 8 GPUs freed, plus LeaderWorkerSet and
  Gateway API CRDs on the cluster, and a dedicated `serveit-cache` PVC. Note that guidellm
  traffic routes through the Gateway/EPP in that path, adding a hop relative to the direct-to-
  vLLM numbers here.
- **Four intermediate workload shapes** between 512/256 and 14000/5000.
- **RHAIIS image comparison.** All results here are upstream vLLM 0.29.0. The registry.redhat.io
  RHAIIS images require a Customer Portal service account for pull; not set up.
