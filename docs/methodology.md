# Methodology

## Environment

| item | value |
|---|---|
| Hardware | 1 node, 8x NVIDIA H200 (141 GB each), NVLink |
| Platform | OpenShift, single pod with 8 GPUs, 64Gi /dev/shm |
| Engine | `vllm/vllm-openai:latest`, vLLM 0.29.0, torch 2.13.0+cu130 |
| Model | `RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-dynamic` |
| Parallelism | TP8 with `--enable-expert-parallel` |
| Storage | model weights (~530 GB) and workspace on separate RWO PVCs |
| Benchmark | guidellm, `synthetic_text` data source |

Model characteristics relevant to tuning: hybrid Mamba/attention/MoE, 128 layers
(16 attention, 64 Mamba, 48 MoE), 512 experts with top-22 routing.

Reported at startup: ~70.5 GiB of weights per GPU (72.0 GiB with MTP), KV cache pool of
14,237,538 tokens at `--max-num-seqs 320`.

## Why a single aggregated TP8 replica

On one 8xH200 node there is only one viable topology:

- ~530 GB of FP8 weights do not fit below TP8 on 141 GB cards.
- Prefill/decode disaggregation and multi-replica expert parallel both require >= 2 pods.
- The model's Mamba `n_groups=8` restricts TP to {1, 2, 4, 8}.

So this study is a flag and concurrency sweep on a fixed topology, not an architecture search.

## Procedure

For each configuration:

1. Stop any running server, confirm all 8 GPUs are released (`nvidia-smi`).
2. Start the server (`scripts/serve-*.sh`), wait for `/v1/models` to respond.
3. Record `GPU KV cache size` and `Maximum concurrency` from the startup log.
4. Run the concurrency sweep (`scripts/bench.sh`).
5. Archive the guidellm JSON.

Weights are loaded from a PVC and stay in page cache across restarts, so only the first
model load is slow.

## Sweep parameters

| shape | ISL | OSL | concurrency levels | duration per level |
|---|---|---|---|---|
| short | 512 | 256 | 8, 16, 32, 64, 128, 200, 300 | 300 s |
| long | 14,000 | 5,000 | 8, 32, 128, 300 | 900 s |

guidellm's synthetic generator emits 528 and 14,016 input tokens respectively (nearest
tokenizer boundary).

## Reading the output

- `gen tok/s` in the benchmark panel counts **completed requests only**. At high concurrency on
  the long shape a large fraction of requests are still streaming when the window closes, so
  compare it against the all-requests throughput table before drawing conclusions.
- Check the completed-request count before trusting p95/p99 on the long shape.
- `Stream Iter Per Req` is how MTP acceptance was derived (see `results/analysis.md`).

## Reproducing

```bash
oc apply -f scripts/pod.yaml
oc rsh nemotron-bench

# inside the pod
mkdir -p /workspace/{logs,results}
hf download RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-dynamic

./serve-tuned.sh &
until curl -s http://127.0.0.1:8000/v1/models >/dev/null; do sleep 15; done
./bench.sh short tuned
./bench.sh long  tuned
```

## Known issues encountered

- `guidellm benchmark` does not exist in the current CLI; the command is `guidellm run`.
- `--constraint kind=max_duration` requires `seconds=`, not `value=`.
- `HF_HUB_ENABLE_HF_TRANSFER` is deprecated in huggingface-hub 1.32; use
  `HF_XET_HIGH_PERFORMANCE=1`.
- Running two concurrent `hf download` processes against the same cache corrupts progress;
  kill all but one and let it resume.
- vLLM exits on unknown flags. Verify Mamba and speculative-decoding flags exist in the target
  image before committing GPU time:
  `vllm serve --help=all | grep -E -- "--mamba-cache-mode|--speculative-config|--moe-backend"`
