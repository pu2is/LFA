"""Redis pub/sub helpers for real-time event streaming.

Infrastructure layer (like queue.py). Business modules call publish_event()
to broadcast status changes; the events module subscribes and streams to
clients via SSE.

publish_event() uses a sync Redis client (called from RQ workers).
async_subscribe_events() uses redis.asyncio for the SSE endpoint, giving
deterministic cleanup when the client disconnects.
"""

import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from app.shared.config import settings

logger = logging.getLogger(__name__)

CHANNEL = "lfa:events"

_redis = Redis.from_url(settings.redis_url)


def publish_event(event_type: str, data: dict[str, Any]) -> None:
    """Publish an event to the Redis channel. Fire-and-forget."""
    try:
        payload = json.dumps(
            {"event": event_type, "data": data, "timestamp": datetime.now(timezone.utc).isoformat()},
        )
        _redis.publish(CHANNEL, payload)
    except Exception:
        logger.debug("Failed to publish event (Redis may be unavailable)", exc_info=True)


async def async_subscribe_events() -> AsyncGenerator[dict[str, Any], None]:
    """Async subscribe to the Redis channel and yield parsed events.

    Uses redis.asyncio so the generator runs in the event loop, not a
    threadpool thread. When the SSE client disconnects, EventSourceResponse
    cancels the task, triggering the finally block for deterministic cleanup.
    Keepalive is handled by EventSourceResponse's ping parameter, not here.
    """
    conn = AsyncRedis.from_url(settings.redis_url, socket_timeout=None)
    pubsub = conn.pubsub()
    await pubsub.subscribe(CHANNEL)

    try:
        while True:
            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=5.0,
            )
            if message is None:
                continue
            if message["type"] != "message":
                continue
            try:
                yield json.loads(message["data"])
            except (json.JSONDecodeError, TypeError):
                continue
    finally:
        await pubsub.unsubscribe(CHANNEL)
        await pubsub.aclose()
        await conn.aclose()
