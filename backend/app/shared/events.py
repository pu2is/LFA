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
from typing import TYPE_CHECKING, Any

from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from app.shared.config import settings

if TYPE_CHECKING:
    from app.modules.jobs.models import Job

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


def publish_job_status(job: "Job", **extra: Any) -> None:
    """Publish a job_status event with the payload shape shared by all job types.

    Fields are included only when set on the job, so scan jobs (path_id, no
    file_id) and file-scoped jobs (file_id, no path_id) both get a payload
    with just the fields that apply to them. Callers can add job-specific
    fields (e.g. file_count) via **extra.
    """
    data: dict[str, Any] = {
        "job_id": str(job.id),
        "type": job.type,
        "status": job.status,
    }
    if job.path_id:
        data["path_id"] = str(job.path_id)
    if job.file_id:
        data["file_id"] = str(job.file_id)
    if job.stage:
        data["stage"] = job.stage
    if job.mode:
        data["mode"] = job.mode
    if job.error_message:
        data["error_message"] = job.error_message
    data.update(extra)
    publish_event("job_status", data)


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
