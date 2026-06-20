"""Redis pub/sub helpers for real-time event streaming.

Infrastructure layer (like queue.py). Business modules call publish_event()
to broadcast status changes; the events module subscribes and streams to
clients via SSE.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Generator

from redis import Redis

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


def subscribe_events() -> Generator[dict[str, Any] | None, None, None]:
    """Subscribe to the Redis channel and yield parsed events.

    Uses get_message(timeout=30) instead of listen() to avoid blocking
    forever — when no message arrives within the timeout window, yields
    None so the caller can send an SSE keepalive comment.
    """
    conn = Redis.from_url(settings.redis_url, socket_timeout=None)
    pubsub = conn.pubsub()
    pubsub.subscribe(CHANNEL)

    try:
        while True:
            message = pubsub.get_message(timeout=30)
            if message is None:
                yield None
                continue
            if message["type"] != "message":
                continue
            try:
                yield json.loads(message["data"])
            except (json.JSONDecodeError, TypeError):
                continue
    finally:
        pubsub.unsubscribe(CHANNEL)
        pubsub.close()
        conn.close()
