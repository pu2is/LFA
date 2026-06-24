from redis import Redis
from rq import Queue

from app.shared.config import settings


redis_connection = Redis.from_url(settings.redis_url)

scan_queue = Queue(
    name="scan",
    connection=redis_connection,
)

ingest_queue = Queue(
    name="ingest",
    connection=redis_connection,
)

label_queue = Queue(
    name="label",
    connection=redis_connection,
)

embed_queue = Queue(
    name="embed",
    connection=redis_connection,
)
