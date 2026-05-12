from celery import Celery
from backend.config import settings

celery_app = Celery(
    "statscout_video",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["backend.video.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    broker_heartbeat=0,
    broker_connection_retry_on_startup=True,
    worker_cancel_long_running_tasks_on_connection_loss=False,
)
