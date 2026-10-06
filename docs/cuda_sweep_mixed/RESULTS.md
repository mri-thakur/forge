# Measured scheduling results

Backend: `torch` / `cuda`. Parameters: 14,260,608. Workload: `mixed`.

Each cell is the median across saved seeds. TTFT includes queue time. Throughput and goodput include drain time. Rejected requests remain in the SLO-fraction denominator. These are in-process results, not HTTP server results.

| Offered req/s | Policy | Repeats | Output tok/s | p99 TTFT ms | SLO fraction | Goodput req/s |
|---:|---|---:|---:|---:|---:|---:|
| 10 | chunked | 3 | 183.0 | 159.0 | 99.7% | 9.37 |
| 10 | continuous | 3 | 183.0 | 56.2 | 100.0% | 9.54 |
| 10 | static | 3 | 183.0 | 392.7 | 99.9% | 9.58 |
| 20 | chunked | 3 | 364.9 | 206.2 | 100.0% | 19.13 |
| 20 | continuous | 3 | 365.3 | 134.8 | 100.0% | 19.14 |
| 20 | static | 3 | 363.2 | 525.1 | 98.6% | 18.94 |
| 25 | chunked | 3 | 454.2 | 189.5 | 100.0% | 23.80 |
| 25 | continuous | 3 | 455.8 | 137.2 | 100.0% | 23.89 |
| 25 | static | 3 | 451.9 | 711.1 | 89.9% | 22.96 |
| 30 | chunked | 3 | 543.6 | 548.6 | 98.5% | 27.89 |
| 30 | continuous | 3 | 544.9 | 391.9 | 100.0% | 28.56 |
| 30 | static | 3 | 472.1 | 4789.1 | 18.2% | 4.62 |
| 35 | chunked | 3 | 626.6 | 881.9 | 81.9% | 26.36 |
| 35 | continuous | 3 | 619.1 | 685.9 | 95.1% | 30.63 |
| 35 | static | 3 | 489.1 | 8823.1 | 3.3% | 0.85 |

With small traces, p99 is descriptive and statistically uncertain. Use larger request counts and multiple independent repetitions before making deployment-capacity or resume claims.
