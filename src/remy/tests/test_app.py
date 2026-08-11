"""Tests for the FastAPI app endpoints."""

from io import StringIO

from fastapi.testclient import TestClient

from remy.app import RecipeAgent, app, log


def test_stream_recipe_agent_endpoint_emits_progress_events(mocker):
    """The streaming endpoint should return SSE progress events."""

    class MockAgent:
        async def stream(self, user_request: str):
            yield {"type": "progress", "message": "Understanding your request..."}
            yield {"type": "result", "payload": {"user_request": user_request}}

    mocker.patch("remy.app.RecipeAgent.from_env", return_value=MockAgent())

    client = TestClient(app)
    response = client.post("/api/recipes/stream", json={"user_request": "Make me a salad"})

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "event: progress" in response.text
    assert "Understanding your request..." in response.text
    assert "event: result" in response.text


def test_stream_recipe_agent_endpoint_logs_stream_error_events(mocker):
    """The streaming endpoint should log error events emitted by the agent."""

    class MockAgent:
        async def stream(self, user_request: str):
            yield {"type": "error", "message": f"Could not process: {user_request}"}

    mocker.patch.object(RecipeAgent, "from_env", return_value=MockAgent())

    output = StringIO()
    sink_id = log.add(output, level="ERROR")

    try:
        client = TestClient(app)
        response = client.post("/api/recipes/stream", json={"user_request": "broken url"})
    finally:
        log.remove(sink_id)

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "Recipe agent emitted error event for request" in output.getvalue()


def test_stream_recipe_agent_endpoint_logs_stream_exceptions(mocker):
    """The streaming endpoint should log exceptions raised by the agent stream."""

    class MockAgent:
        async def stream(self, user_request: str):
            raise RuntimeError(f"agent crashed for {user_request}")
            yield  # pragma: no cover - keeps this as an async generator

    mocker.patch.object(RecipeAgent, "from_env", return_value=MockAgent())

    output = StringIO()
    sink_id = log.add(output, level="ERROR")

    try:
        client = TestClient(app)
        response = client.post("/api/recipes/stream", json={"user_request": "bad prompt"})
    finally:
        log.remove(sink_id)

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "agent crashed for bad prompt" in response.text
    assert "Recipe agent stream failed for request" in output.getvalue()


def test_stream_recipe_agent_endpoint_serializes_array_like_payload_values(mocker):
    """The streaming endpoint should serialize array-like values in SSE payloads."""

    class ArrayLike:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def tolist(self) -> list[float]:
            return self._values

    class MockAgent:
        async def stream(self, user_request: str):
            yield {
                "type": "result",
                "payload": {
                    "user_request": user_request,
                    "recipe": {"instru_embedding": ArrayLike([0.1, 0.2, 0.3])},
                },
            }

    mocker.patch.object(RecipeAgent, "from_env", return_value=MockAgent())

    client = TestClient(app)
    response = client.post("/api/recipes/stream", json={"user_request": "test array"})

    assert response.status_code == 200
    assert "event: result" in response.text
    assert '"instru_embedding": [0.1, 0.2, 0.3]' in response.text
    assert "event: error" not in response.text
