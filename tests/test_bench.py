import numpy as np

from forge.bench import make_trace, run_trace, summarize
from forge.engine import Request
from forge.model import ModelConfig, NumpyModel


def test_trace_is_reproducible_and_arrivals_are_independent():
    a = make_trace(30, 20, seed=7)
    assert a == make_trace(30, 20, seed=7)
    assert a != make_trace(30, 20, seed=8)
    assert all(x["arrival"] <= y["arrival"] for x, y in zip(a, a[1:]))


def test_goodput_denominator_counts_rejections_and_drain():
    completed = Request(
        "a", [1], 2, 0, generated=[2, 3], emissions=[0.1, 0.12], finished=0.12, status="completed"
    )
    slow = Request(
        "b", [1], 2, 0, generated=[2, 3], emissions=[1, 1.02], finished=1.02, status="completed"
    )
    rejected = Request("c", [1], 2, 0, status="rejected")
    result = summarize([completed, slow, rejected], 2.0, 0.5, 0.05)
    assert result["slo_requests"] == 1
    assert result["slo_fraction_all_submissions"] == 1 / 3
    assert result["slo_goodput_requests_per_second"] == 0.5
    assert result["output_tokens_per_second"] == 2


def test_open_loop_runner_drains_and_records_every_request():
    model = NumpyModel(ModelConfig(dim=8, heads=2, kv_heads=1, layers=1, hidden=16))
    trace = [{"id": str(i), "arrival": 0, "prompt": [1, 2], "max_tokens": 3} for i in range(4)]
    result = run_trace(model, trace, "chunked", blocks=8, max_batch=2, token_budget=4)
    assert result["submitted"] == result["completed"] == 4
    assert result["cache"]["used_blocks"] == 0
    assert result["output_tokens"] == 12
    assert np.isfinite(result["output_tokens_per_second"])
