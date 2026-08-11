import json
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from remy.agents.recipe_agent import RecipeAgent

app = FastAPI(title="Remy", version="0.1.0")


class RecipeStreamRequest(BaseModel):
    """Request payload for streaming recipe extraction."""

    user_request: str


@app.get("/")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/recipes/stream")
async def stream_recipe_agent(request: RecipeStreamRequest) -> StreamingResponse:
    """Stream recipe-agent progress updates as SSE events."""

    agent = RecipeAgent.from_env()

    async def event_generator() -> Any:
        try:
            async for event in agent.stream(request.user_request):
                yield _format_sse_event(event)
        except Exception as exc:  # pragma: no cover - defensive fallback
            yield _format_sse_event({"type": "error", "message": str(exc)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _format_sse_event(event: dict[str, Any]) -> str:
    """Format a payload as a Server-Sent Event."""

    event_name = event.get("type", "message")
    payload = json.dumps(event)
    return f"event: {event_name}\ndata: {payload}\n\n"
