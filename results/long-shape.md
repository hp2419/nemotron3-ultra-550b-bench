# Long shape — 14,000 in / 5,000 out

Synthetic text, fixed lengths. 900 s per concurrency level. Single 8xH200 node, TP8.

guidellm's synthetic generator produced 14,016 input tokens per request (nearest tokenizer
boundary), so total context is 19,016 tokens per request.

Medians unless noted.

## All three configurations

| c | metric | baseline (128 seqs) | MTP k=2 (128 seqs) | MTP k=2 + 320 seqs |
|---|---|---|---|---|
| 8 | TTFT | 3.26 s | 1.06 s | 1.06 s |
| | ITL | 12.4 ms | 6.7 ms | 6.7 ms |
| | E2E | 65.4 s | 34.7 s | 34.8 s |
| | gen tok/s | 582 | 1,116 | 1,116 |
| | completed | 105 | 201 | 201 |
| 32 | TTFT | 3.28 s | 2.25 s | 2.41 s |
| | ITL | 21.6 ms | 13.6 ms | 13.6 ms |
| | E2E | 113.7 s | 70.3 s | 70.6 s |
| | gen tok/s | 1,261 | 2,138 | 2,137 |
| | completed | 227 | 385 | 385 |
| 128 | TTFT | 3.35 s (p95 95.7 s) | 5.56 s (p95 97 s) | 7.33 s (p95 97.5 s) |
| | ITL | 50.9 ms | 40.6 ms | 40.8 ms |
| | E2E | 258.1 s | 210 s | 212.2 s |
| | gen tok/s | 2,136 | 2,533 | 2,500 |
| | completed | 385 | 385 | 450 |
| 300 | TTFT | 298.9 s | 260 s | 134.7 s |
| | ITL | 50.5 ms | 41.1 ms | 85.9 ms |
| | E2E | 551.6 s | 465 s | 565.1 s |
| | gen tok/s | 2,136 | 2,461 | 1,717 |
| | completed | 385 | 385 | 309 |

## Findings

**1. `--max-num-seqs 320` provides no benefit on this shape.**
Below 128 concurrent the cap never binds, so the c=8/32/128 columns are identical to the
MTP-at-128 run within noise.

**2. At c=300 it is actively harmful.**
Each request carries 19,016 tokens, so admitting 300 of them requires roughly 4.2M tokens of
prefill work. At `--max-num-batched-tokens 8192` that is ~513 prefill steps competing with
decode. ITL doubles (41 -> 86 ms) and E2E reaches 565 s.

Caveat on the 1,717 tok/s figure: it counts completed requests only. The all-requests view shows
raw output throughput actually rose slightly (2,841 -> 2,991 tok/s) because 299 requests were
still mid-stream when the 900 s window closed. So the drop is partly a measurement artifact of
the window and partly a genuine latency collapse. Either way c=300 is not a usable operating
point on this shape with one node.

**3. Baseline saturation at c=128 is the flag, not the hardware.**
Baseline c=128 and c=300 are identical to four significant figures: 2,136.2 vs 2,136.4 gen tok/s,
50.9 vs 50.5 ms ITL, and 385 completed in both. Step time did not change when 172 more requests
were offered, which can only happen if the running batch size did not change.

**4. MTP is a larger win here than on the short shape.**
At c=8: -46% ITL, +92% output throughput, -47% E2E versus baseline.

**5. Recommended operating point is around c=32.**
2.41 s TTFT, 13.6 ms ITL, 2,137 gen tok/s — 86% of the peak observed throughput at roughly a
third of the latency of c=128. Moving to c=128 buys +17% throughput for 3x the ITL and 3x the
TTFT, which is a poor trade under an interactive SLA.

**6. TTFT spread at c=128 is severe.** Median 3.3-7.3 s against p95 of 96-98 s. Under an
interactive SLA the p95 is the number that matters.

## MTP acceptance on this shape

1,696 stream iterations (median) per 5,000-token request:

```
5000 / 1696                = 2.95 output tokens per iteration
2.95 - 1                   = 1.95 accepted drafts
1.95 / 2                   = ~97% acceptance
```

**This figure is not credible as a workload result.** Random-token prompts produce
low-entropy continuations that the draft head predicts almost perfectly. Do not quote it.
The measured latency and throughput gains are real; the mechanism behind their size is not
representative.

## Unresolved

At c=8 the baseline TTFT is 3.26 s while both MTP runs show 1.06 s. MTP should not affect
prefill. Candidate explanations: differing prefix-cache hit rates between runs, or a prefill
path change from `--mamba-cache-mode align` (required with MTP + chunked prefill). Needs an
isolated A/B before any TTFT improvement is attributed to MTP.
