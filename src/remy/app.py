import json
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from loguru import logger as log
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
                if _is_error_event(event):
                    log.error(
                        "Recipe agent emitted error event for request: {user_request}; event={event}",
                        user_request=request.user_request,
                        event=event,
                    )
                yield _format_sse_event(event)
        except Exception as exc:  # pragma: no cover - defensive fallback
            log.exception(
                "Recipe agent stream failed for request: {user_request}; exception={exception}",
                user_request=request.user_request,
                exception=str(exc),
            )
            yield _format_sse_event({"type": "error", "message": str(exc)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _format_sse_event(event: dict[str, Any]) -> str:
    """Format a payload as a Server-Sent Event.

    Args:
        event: Event payload to stream to the client.

    Returns:
        SSE-formatted event string.
    """

    event_name = event.get("type", "message")
    payload = json.dumps(event, default=_json_serializer)
    return f"event: {event_name}\ndata: {payload}\n\n"


def _is_error_event(event: dict[str, Any]) -> bool:
    """Return whether an SSE payload represents an error event.

    Args:
        event: A streamed recipe agent event payload.

    Returns:
        True when the payload indicates an error, otherwise False.
    """

    event_type = event.get("type")
    return isinstance(event_type, str) and event_type.lower() == "error"


def _json_serializer(value: Any) -> Any:
    """Serialize unsupported JSON values used in streamed events.

    Args:
        value: Object that cannot be serialized by ``json.dumps`` directly.

    Returns:
        JSON-compatible representation.
    """

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")

    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return to_list()

    return str(value)
