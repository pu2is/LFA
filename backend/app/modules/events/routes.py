"""SSE endpoint for real-time event streaming to the frontend.

Subscribes to the shared Redis pub/sub channel and yields events in
Server-Sent Events format over a long-lived HTTP response.
"""

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.shared.events import subscribe_events

router = APIRouter(prefix="/events", tags=["events"])


def _sse_stream():
    """Synchronous generator that yields SSE-formatted lines.

    subscribe_events() yields None on timeout (no message within 30s).
    We send an SSE comment (: keepalive) to prevent proxies and browsers
    from closing the idle connection.
    """
    for event in subscribe_events():
        if event is None:
            yield ": keepalive\n\n"
            continue
        event_type = event.get("event", "message")
        data_json = json.dumps(event.get("data", {}))
        yield f"event: {event_type}\ndata: {data_json}\n\n"


@router.get("/stream")
def event_stream():
    return StreamingResponse(
        _sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
