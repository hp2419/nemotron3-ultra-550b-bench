# Analysis — why the numbers came out this way

## 1. Proving the batch was capped, not the GPU saturated

The baseline config showed identical behaviour at 128 and 300 concurrent:

```
long shape         c=128      c=300
ITL                50.9 ms    50.5 ms
gen tok/s          2136.2     2136.4
completed (900s)   385        385
```

If the GPU were compute-saturated, offering 172 more requests would lengthen each decode step.
Step time did not move at all. The only configuration in which step time is invariant to offered
load is one where the running batch size is invariant — i.e. pinned at `--max-num-seqs 128`.

Cross-check with Little's law (`L = lambda * W`):

```
c=300 baseline, long shape:
  lambda = 0.4 req/s,  W = 551.6 s
  L      = 0.4 * 551.6 = 221 requests in the system
  running = 128 (the cap)
  queued  = 221 - 128 = 93
```

## 2. Decomposing TTFT

`TTFT = queue wait + prefill time`

At c=8 nothing queues, so TTFT is essentially prefill: 397 ms on the short shape.

At c=300 with a 128-slot cap, 172 requests wait. With a completion rate of 10.2 req/s:

```
predicted queue wait = 172 / 10.2 = 16.9 s
measured TTFT        = 14.96 s
```

Within ~13%, which is close enough given median-vs-mean spread. The 15-second first-token
latency was a queueing artifact, not model speed.

With 320 slots all 300 requests are admitted, queue wait goes to ~0, and TTFT drops to 1.6 s —
consistent with prefill alone (300 * 528 = 158k tokens at 8,192 batched tokens/step ~= 19 steps).

## 3. Why a larger batch raises throughput (sublinearly)

`tokens/s = batch / step_time`. From the 320-seq run on the short shape:

```
c=128:  128 / 0.0440 = 2,909 tok/s
c=200:  200 / 0.0617 = 3,241 tok/s
c=300:  300 / 0.0876 = 3,425 tok/s    (+18% over c=128)
```

Per-sequence step cost falls: 0.344 -> 0.309 -> 0.292 ms. Some component of step time is fixed
rather than per-sequence.

The likely mechanism is MoE weight reading. The model has 512 experts with top-22 routing. At
batch 128 that is 128 * 22 = 2,816 expert activations spread across 512 experts, so nearly every
expert is touched and effectively the entire expert set is read from HBM. At batch 300 the same
weights are read and 2.3x as many tokens are produced from them.

Rough roofline on the expert read (order-of-magnitude, from published parameter counts rather
than measured): ~520 GB of expert weights in FP8 across 8 GPUs is ~65 GB/GPU; at 4.8 TB/s HBM
that is ~13.5 ms. Measured step time at c=128 is ~44 ms, so roughly 3x of the step is attention,
Mamba, all-to-all, activations and launch overhead. Those components do scale with batch, which
is why the gain is +18% rather than the +134% a pure weight-read model would predict.

This estimate should be validated with an nsys profile before being relied on.

## 4. MTP arithmetic

Acceptance rate from iteration counts:

```
short shape (512/256):  256 / 108  = 2.37 tok/iter -> (2.37-1)/2 = 68%
long shape (14k/5k):   5000 / 1696 = 2.95 tok/iter -> (2.95-1)/2 = 97%
no MTP:                 256 / 254  = 1.00 tok/iter
```

MTP performs ~2.4x fewer forward passes, but each pass verifies 3 tokens instead of 1, so the
net gain depends on how much headroom the GPU has:

```
short shape, c=8:    553 -> 699 tok/s   = +26%   (GPU underutilised, extra work nearly free)
short shape, c=300: 3,073 -> 3,328      = +8%    (GPU busy, extra work has a real cost)
long shape,  c=8:    582 -> 1,116       = +92%
```

The 97% figure on the long shape is a property of the synthetic data, not of the model or
workload. See the caveat in `results/long-shape.md`.

## 5. Concurrency ceilings

Three distinct limits, often conflated:

**Memory.** vLLM reported a KV pool of 14,237,538 tokens at `--max-num-seqs 320`, and
"Maximum concurrency for 64,000 tokens per request: 222.46x".

| shape | tokens/req | KV-limited requests |
|---|---|---|
| 512/256 | 768 | ~18,500 |
| 14k/5k | 19,016 | ~750 |
| full 64k context | 64,000 | 222 |

On the short shape KV is not the binding pool: Mamba state is roughly 33 MiB/seq/GPU (derived
from the HF config, not measured), which would exhaust memory somewhere around 1,500-2,000
sequences regardless of prompt length.

**Configuration.** `--max-num-seqs`. This is what actually bound the stock config.

**Latency.** The one that matters operationally. On 512/256 with the tuned config:

| concurrency | ITL | TTFT |
|---|---|---|
| 128 | 43 ms | 0.54 s |
| 200 | 65 ms | 0.82 s |
| 300 | 82 ms | 1.18 s |

An ITL target of 50 ms implies roughly 150 concurrent; 80 ms implies ~290. A TTFT-only target of
2 s admits 300+. Picking an operating point requires an SLA.
