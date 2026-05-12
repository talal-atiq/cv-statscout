import uuid
import os
import hashlib
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse
from backend.database import jobs_collection
from backend.video.schemas import UploadResponse, StatusResponse, JobStatus, DemoSummary
from backend.video.tasks import process_video_task
from backend.config import settings

router = APIRouter(prefix="/api/video", tags=["video"])

CHUNK_SIZE = 1024 * 1024  # 1 MB


@router.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    Streams the upload to disk while computing SHA-256.
    If a completed job with the same hash already exists, returns it (cache hit).
    Otherwise creates a new job and enqueues processing.
    """
    allowed_types = ["video/mp4", "video/x-msvideo", "video/quicktime", "video/x-matroska"]
    if file.content_type not in allowed_types:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")

    job_id = str(uuid.uuid4())
    os.makedirs(settings.upload_dir, exist_ok=True)
    ext = os.path.splitext(file.filename)[1]
    save_path = os.path.join(settings.upload_dir, f"{job_id}{ext}")

    hasher = hashlib.sha256()
    with open(save_path, "wb") as buffer:
        while chunk := await file.read(CHUNK_SIZE):
            hasher.update(chunk)
            buffer.write(chunk)
    video_hash = hasher.hexdigest()

    # Cache lookup: any completed job with the same content?
    cached = await jobs_collection.find_one(
        {"video_hash": video_hash, "status": JobStatus.DONE.value},
        {"_id": 0},
    )
    if cached:
        try:
            os.remove(save_path)  # discard duplicate upload
        except OSError:
            pass
        return UploadResponse(
            job_id=cached["job_id"],
            message="Cached result returned (instant).",
        )

    now = datetime.now(timezone.utc)
    job_doc = {
        "job_id": job_id,
        "video_hash": video_hash,
        "status": JobStatus.QUEUED.value,
        "progress": 0,
        "status_message": "Video queued for processing",
        "created_at": now,
        "updated_at": now,
        "video_filename": file.filename,
        "video_path": save_path,
        "total_frames": 0,
        "processed_frames": 0,
        "output_video_path": None,
        "analytics": None,
        "error": None,
    }
    await jobs_collection.insert_one(job_doc)

    process_video_task.delay(job_id, save_path)

    return UploadResponse(job_id=job_id, message="Video uploaded. Processing started.")


def _build_video_urls(job: dict, job_id: str) -> dict:
    """Return main/pitch/voronoi/output URLs based on what's present on disk.

    New jobs have separate main/pitch/voronoi files. Legacy jobs only have
    output_video_path — exposed as both output_video_url and main_video_url.
    """
    urls = {
        "output_video_url": None,
        "main_video_url": None,
        "pitch_video_url": None,
        "voronoi_video_url": None,
    }
    if job.get("main_video_path") and os.path.exists(job["main_video_path"]):
        urls["main_video_url"] = f"/api/video/stream/{job_id}/main"
        urls["output_video_url"] = urls["main_video_url"]
    elif job.get("output_video_path") and os.path.exists(job["output_video_path"]):
        urls["output_video_url"] = f"/api/video/stream/{job_id}"
        urls["main_video_url"] = urls["output_video_url"]
    if job.get("pitch_video_path") and os.path.exists(job["pitch_video_path"]):
        urls["pitch_video_url"] = f"/api/video/stream/{job_id}/pitch"
    if job.get("voronoi_video_path") and os.path.exists(job["voronoi_video_path"]):
        urls["voronoi_video_url"] = f"/api/video/stream/{job_id}/voronoi"
    return urls


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """Frontend polls this every 2 seconds."""
    job = await jobs_collection.find_one({"job_id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(404, "Job not found")

    urls = _build_video_urls(job, job_id)

    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        status_message=job["status_message"],
        analytics=job.get("analytics"),
        error=job.get("error"),
        **urls,
    )


_VIEW_TO_FIELD = {
    "main": "main_video_path",
    "pitch": "pitch_video_path",
    "voronoi": "voronoi_video_path",
}


@router.get("/stream/{job_id}")
async def stream_video_legacy(job_id: str):
    """Legacy single-video stream endpoint — used by jobs processed before the
    separate-panels refactor."""
    job = await jobs_collection.find_one({"job_id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(404, "Job not found")
    path = job.get("main_video_path") or job.get("output_video_path")
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Processed video not found")
    return FileResponse(path, media_type="video/mp4", filename=f"statscout_{job_id}.mp4")


@router.get("/stream/{job_id}/{view}")
async def stream_video_view(job_id: str, view: str):
    """Serves one of the three output panels: main | pitch | voronoi."""
    field = _VIEW_TO_FIELD.get(view)
    if not field:
        raise HTTPException(400, f"Unknown view '{view}'. Must be main/pitch/voronoi.")
    job = await jobs_collection.find_one({"job_id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(404, "Job not found")
    path = job.get(field)
    # Fallback for legacy jobs that only have a hstacked output_video_path
    if not path and view == "main":
        path = job.get("output_video_path")
    if not path or not os.path.exists(path):
        raise HTTPException(404, f"Video for view '{view}' not found")
    return FileResponse(path, media_type="video/mp4", filename=f"statscout_{job_id}_{view}.mp4")


@router.get("/demos", response_model=list[DemoSummary])
async def list_demos():
    """Returns the curated list of pre-baked demo clips surfaced on the landing page."""
    cursor = jobs_collection.find(
        {"is_demo": True, "status": JobStatus.DONE.value},
        {"_id": 0},
    ).sort("demo_order", 1)

    demos = []
    async for job in cursor:
        urls = _build_video_urls(job, job["job_id"])
        demos.append(DemoSummary(
            job_id=job["job_id"],
            title=job.get("demo_title") or job.get("video_filename", "Demo clip"),
            description=job.get("demo_description"),
            output_video_url=urls["output_video_url"] or f"/api/video/stream/{job['job_id']}",
            main_video_url=urls["main_video_url"],
            pitch_video_url=urls["pitch_video_url"],
            voronoi_video_url=urls["voronoi_video_url"],
        ))
    return demos


@router.post("/admin/mark-demo/{job_id}")
async def mark_as_demo(
    job_id: str,
    title: str = Body(..., embed=True),
    description: str | None = Body(None, embed=True),
    order: int = Body(0, embed=True),
):
    """Mark a completed job as a demo clip. Admin-only — gate behind auth in prod."""
    result = await jobs_collection.update_one(
        {"job_id": job_id, "status": JobStatus.DONE.value},
        {"$set": {
            "is_demo": True,
            "demo_title": title,
            "demo_description": description,
            "demo_order": order,
        }},
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Completed job not found")
    return {"job_id": job_id, "is_demo": True}


@router.delete("/admin/mark-demo/{job_id}")
async def unmark_demo(job_id: str):
    """Remove demo status from a job."""
    await jobs_collection.update_one(
        {"job_id": job_id},
        {"$unset": {"is_demo": "", "demo_title": "", "demo_description": "", "demo_order": ""}},
    )
    return {"job_id": job_id, "is_demo": False}
