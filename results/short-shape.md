# Short shape — 512 in / 256 out

Synthetic text, fixed lengths. 300 s per concurrency level. Single 8xH200 node, TP8.

Medians unless noted. `gen tok/s` is output tokens per second over completed requests.

## Config D — `--max-num-seqs 320` + MTP k=2  (best)

| concurrency | req/s | gen tok/s | TTFT | ITL | E2E | completed |
|---|---|---|---|---|---|---|
| 8 | 2.7 | 699 | 166 ms | 10.7 ms | 2.9 s | 819 |
| 16 | 3.9 | 989 | 172 ms | 15.4 ms | 4.1 s | 1,159 |
| 32 | 5.5 | 1,402 | 282 ms | 21.9 ms | 5.8 s | 1,643 |
| 64 | 7.8 | 1,992 | 387 ms | 30.7 ms | 8.2 s | 2,334 |
| 128 | 11.0 | 2,815 | 541 ms | 42.8 ms | 11.5 s | 3,299 |
| 200 | 11.0 | 2,827 | 823 ms | 65.2 ms | 17.5 s | 3,313 |
| 300 | 13.0 | 3,328 | 1,184 ms | 81.8 ms | 22.0 s | 3,900 |

The c=200 row is non-monotonic against c=128 (same req/s, same completed count) and then jumps
at c=300. The 320-only run below is smooth through the same range, so this is most likely an
unlucky 300 s window rather than real behaviour. Flagged for rerun; do not use for a curve.

## Config C — `--max-num-seqs 320`, no MTP

| concurrency | req/s | gen tok/s | TTFT | ITL | E2E | completed |
|---|---|---|---|---|---|---|
| 8 | 2.2 | 553 | 393 ms | 12.8 ms | 3.7 s | 649 |
| 16 | 3.5 | 888 | 638 ms | 15.5 ms | 4.6 s | 1,041 |
| 32 | 5.0 | 1,284 | 664 ms | 21.9 ms | 6.3 s | 1,505 |
| 64 | 7.4 | 1,898 | 1,128 ms | 29.1 ms | 8.5 s | 2,224 |
| 128 | 10.0 | 2,551 | 1,150 ms | 44.3 ms | 12.5 s | 2,990 |
| 200 | 11.3 | 2,901 | 1,563 ms | 61.4 ms | 17.2 s | 3,400 |
| 300 | 12.0 | 3,073 | 1,600 ms | 87.5 ms | 23.9 s | 3,601 |

## Configs A and B — `--max-num-seqs 128`

Baseline (A) and MTP k=2 (B) were run before the raw sweep output was archived in full.
Verified values only; the complete per-level data is in the raw JSON (see `docs/methodology.md`).

| concurrency | A: TTFT | A: gen tok/s | B: TTFT | B: gen tok/s |
|---|---|---|---|---|
| 8 | 397 ms | 553 | 170 ms | 684 |
| 128 | 1,541 ms | 2,593 | 513 ms | 2,751 |
| 300 | 14,964 ms | ~2,593 (flat) | 14,423 ms | ~2,600 (flat) |

Config A saturates at c=128: request rate is flat at 10.2 req/s from 128 onward. That flatness
is the `--max-num-seqs 128` cap, not the hardware — see `results/analysis.md`.

## Comparison at c=300

| | A: stock | B: +MTP | C: +320 seqs | D: both |
|---|---|---|---|---|
| TTFT | 14,964 ms | 14,423 ms | 1,600 ms | **1,184 ms** |
| gen tok/s | ~2,593 | ~2,600 | 3,073 | **3,328** |
| ITL | ~44 ms | ~41 ms | 87.5 ms | **81.8 ms** |
| E2E | ~26 s | ~25 s | 23.9 s | **22.0 s** |
| req/s | 10.2 | 10.4 | 12.0 | **13.0** |

The two changes compose: `max-num-seqs 320` contributes +18%, MTP adds a further +8%.

## MTP acceptance

Config D, all concurrency levels: 108 stream iterations (median) per 256-token request.

```
256 / 108                  = 2.37 output tokens per iteration
2.37 - 1 (guaranteed base) = 1.37 accepted drafts
1.37 / 2 (drafts offered)  = 68% acceptance
```

Without MTP: 254 iterations for 256 tokens = 1.0 token/iteration, as expected.

68% is reproducible across runs on this shape but was measured on random-token prompts;
treat it as provisional until rerun on representative text.
