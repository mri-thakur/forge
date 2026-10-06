import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from forge.model import ModelConfig, NumpyModel  # noqa: E402
from forge.server import create_app  # noqa: E402


def test_server_nonstream_stream_and_bad_context():
    model = NumpyModel(ModelConfig(dim=8, heads=2, kv_heads=1, hidden=16, layers=1, context=32))
    with TestClient(create_app(model)) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/v1/completions", json={"prompt": "ab", "max_tokens": 4})
        assert response.status_code == 200
        assert response.json()["usage"]["completion_tokens"] == 4
        response = client.post(
            "/v1/completions", json={"prompt": "ab", "max_tokens": 4, "stream": True}
        )
        assert "data: [DONE]" in response.text
        assert (
            client.post("/v1/completions", json={"prompt": "x" * 31, "max_tokens": 4}).status_code
            == 400
        )
