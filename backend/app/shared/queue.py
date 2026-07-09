from redis import Redis
from rq import Queue
from rq.job import Retry

from app.shared.config import settings


redis_connection = Redis.from_url(settings.redis_url)

# Shared retry policy for job types where failure is often transient (a locked
# file, Ollama restarting, a dropped DB connection) -- retrying has a real
# chance of succeeding. Applied to ingest, embed, and label jobs.
#
# Scan jobs deliberately do NOT get this: a scan failure (bad/unreadable
# registered path) is almost always deterministic -- retrying would just fail
# the same way 3 more times. Re-scanning is a cheap, explicit user action
# (POST /scans again) once the underlying path issue is fixed.
JOB_RETRY = Retry(max=3, interval=[60, 120, 300])

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
