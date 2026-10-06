# Measured scheduling results

Backend: `numpy` / `cpu`. Parameters: 115,008. Workload: `prefix`.

Each cell is the median across saved seeds. TTFT includes queue time. Throughput and goodput include drain time. Rejected requests remain in the SLO-fraction denominator. These are in-process results, not HTTP server results.

| Offered req/s | Policy | Repeats | Output tok/s | p99 TTFT ms | SLO fraction | Goodput req/s |
|---:|---|---:|---:|---:|---:|---:|
| 100 | chunked | 3 | 1630.4 | 4.5 | 100.0% | 85.81 |

With small traces, p99 is descriptive and statistically uncertain. Use larger request counts and multiple independent repetitions before making deployment-capacity or resume claims.
