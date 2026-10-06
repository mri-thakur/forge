# Measured scheduling results

Backend: `numpy` / `cpu`. Parameters: 115,008. Workload: `head_of_line`.

Each cell is the median across saved seeds. TTFT includes queue time. Throughput and goodput include drain time. Rejected requests remain in the SLO-fraction denominator. These are in-process results, not HTTP server results.

| Offered req/s | Policy | Repeats | Output tok/s | p99 TTFT ms | SLO fraction | Goodput req/s |
|---:|---|---:|---:|---:|---:|---:|
| 50 | chunked | 3 | 267.5 | 5829.0 | 7.8% | 1.33 |
| 50 | continuous | 3 | 220.1 | 7647.6 | 5.5% | 0.68 |
| 50 | static | 3 | 299.1 | 5613.7 | 7.8% | 1.28 |
| 150 | chunked | 3 | 316.6 | 6225.5 | 7.8% | 1.41 |
| 150 | continuous | 3 | 297.4 | 6924.0 | 6.2% | 1.04 |
| 150 | static | 3 | 312.5 | 6453.0 | 0.8% | 0.14 |

With small traces, p99 is descriptive and statistically uncertain. Use larger request counts and multiple independent repetitions before making deployment-capacity or resume claims.
