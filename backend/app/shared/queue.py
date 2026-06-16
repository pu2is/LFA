from redis import Redis
from rq import Queue

from app.shared.config import settings


redis_connection = Redis.from_url(settings.redis_url)

# check update
scan_queue = Queue(
    name="scan",
    connection=redis_connection,
)

# process: ocr -> chunk -> labeling
labeling_queue = Queue(
    name="labeling",
    connection=redis_connection,
)

# Job2: embed chunks via bge-m3 (lower priority — workers prefer labeling_queue)
embedding_queue = Queue(
    name="embedding",
    connection=redis_connection,
)
