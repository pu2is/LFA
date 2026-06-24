"""SSE endpoint for real-time event streaming to the frontend.

Contract (snapshot + stream):
  1. Client fetches GET /files (or any list endpoint) for the current snapshot.
  2. Client subscribes to GET /events/stream for incremental SSE deltas.
  3. DB is the source of truth — if a client misses events (reconnect gap),
     it re-fetches the snapshot; no server-side replay is needed.
"""

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from app.shared.events import async_subscribe_events

router = APIRouter(prefix="/events", tags=["events"])


async def _event_generator() -> AsyncGenerator[ServerSentEvent, None]:
    async for event in async_subscribe_events():
        event_type = event.get("event", "message")
        data_json = json.dumps(event.get("data", {}))
        yield ServerSentEvent(data=data_json, event=event_type)


@router.get("/stream")
async def event_stream():
    return EventSourceResponse(
        _event_generator(),
        ping=30,
    )
