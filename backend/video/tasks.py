import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from celery import Task
from celery.signals import worker_ready
from datetime import datetime, timezone
import pymongo

from backend.celery_app import celery_app
from backend.config import settings
from backend.video.team_classifier import TeamClassifier  # torch must load before cv2
from backend.video.processor import process_video

_team_classifier: TeamClassifier | None = None


@worker_ready.connect
def preload_models(**kwargs):
    global _team_classifier
    _team_classifier = TeamClassifier()

# Celery workers can't use FastAPI's dependency injection, so we build a
# direct motor client here for status updates.
_mongo_client = pymongo.MongoClient(settings.mongo_uri)
_db = _mongo_client[settings.mongo_db_name]
_jobs = _db["video_jobs"]


def _sync_update(job_id: str, update_dict: dict):
    """Synchronous MongoDB update used inside the Celery worker."""
    update_dict["updated_at"] = datetime.now(timezone.utc)
    _jobs.update_one({"job_id": job_id}, {"$set": update_dict})


@celery_app.task(bind=True, name="process_video")
def process_video_task(self: Task, job_id: str, video_path: str):
    """
    Celery task that wraps the full video processing pipeline.
    Updates the MongoDB job document at each pipeline stage.
    """

    def progress_callback(status: str, progress: int, message: str):
        _sync_update(job_id, {
            "status": status,
            "progress": progress,
            "status_message": message,
        })

    try:
        result = process_video(
            job_id=job_id,
            video_path=video_path,
            progress_callback=progress_callback,
            team_classifier=_team_classifier,
        )
        _sync_update(job_id, {
            "status": "done",
            "progress": 100,
            "status_message": "Processing complete",
            "output_video_path": result["output_video_path"],
            "main_video_path": result.get("main_video_path"),
            "pitch_video_path": result.get("pitch_video_path"),
            "voronoi_video_path": result.get("voronoi_video_path"),
            "analytics": result["analytics"],
        })
    except Exception as e:
        import traceback
        _sync_update(job_id, {
            "status": "failed",
            "progress": 0,
            "status_message": "Processing failed",
            "error": str(e) + "\n" + traceback.format_exc(),
        })
        raise
