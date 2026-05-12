from motor.motor_asyncio import AsyncIOMotorClient
from backend.config import settings

client = AsyncIOMotorClient(settings.mongo_uri)
db = client[settings.mongo_db_name]

# Collections
jobs_collection = db["video_jobs"]


async def ensure_indexes():
    """Create indexes used for caching and lookups. Idempotent."""
    await jobs_collection.create_index("job_id", unique=True)
    await jobs_collection.create_index("video_hash")
    await jobs_collection.create_index([("is_demo", 1), ("demo_order", 1)])
