import json
import os
import hashlib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="StatScout Expo Demo API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the seed data into memory
SEED_FILE = "seed_data.json"
try:
    with open(SEED_FILE, "r", encoding="utf-8") as f:
        jobs_db = json.load(f)
except Exception as e:
    print(f"Failed to load {SEED_FILE}: {e}")
    jobs_db = []

# Schemas
class UploadResponse(BaseModel):
    job_id: str
    message: str

class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    status_message: str
    output_video_url: Optional[str] = None
    main_video_url: Optional[str] = None
    pitch_video_url: Optional[str] = None
    voronoi_video_url: Optional[str] = None
    analytics: Optional[dict] = None
    error: Optional[str] = None

class DemoSummary(BaseModel):
    job_id: str
    title: str
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    output_video_url: str
    main_video_url: Optional[str] = None
    pitch_video_url: Optional[str] = None
    voronoi_video_url: Optional[str] = None

CHUNK_SIZE = 1024 * 1024

def _build_video_urls(job: dict, job_id: str) -> dict:
    urls = {
        "output_video_url": None,
        "main_video_url": None,
        "pitch_video_url": None,
        "voronoi_video_url": None,
    }
    if job.get("main_video_path"):
        urls["main_video_url"] = f"/api/video/stream/{job_id}/main"
        urls["output_video_url"] = urls["main_video_url"]
    elif job.get("output_video_path"):
        urls["output_video_url"] = f"/api/video/stream/{job_id}"
        urls["main_video_url"] = urls["output_video_url"]
    if job.get("pitch_video_path"):
        urls["pitch_video_url"] = f"/api/video/stream/{job_id}/pitch"
    if job.get("voronoi_video_path"):
        urls["voronoi_video_url"] = f"/api/video/stream/{job_id}/voronoi"
    return urls

@app.post("/api/video/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    hasher = hashlib.sha256()
    while chunk := await file.read(CHUNK_SIZE):
        hasher.update(chunk)
    video_hash = hasher.hexdigest()

    # Look for cached job
    cached = next((j for j in jobs_db if j.get("video_hash") == video_hash and j.get("status") == "done"), None)
    if cached:
        return UploadResponse(job_id=cached["job_id"], message="Demo Mode: Cached result returned instantly.")
    
    # If not found, throw error for demo mode
    raise HTTPException(status_code=400, detail="Demo Mode: AI pipeline disabled. Please upload a pre-processed demo video.")

@app.get("/api/video/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    job = next((j for j in jobs_db if j.get("job_id") == job_id), None)
    if not job:
        raise HTTPException(404, "Job not found")
    
    urls = _build_video_urls(job, job_id)
    return StatusResponse(
        job_id=job_id,
        status=job.get("status", "done"),
        progress=job.get("progress", 100),
        status_message=job.get("status_message", "Demo Mode"),
        analytics=job.get("analytics"),
        error=job.get("error"),
        **urls,
    )

@app.get("/api/video/stream/{job_id}")
async def stream_video_legacy(job_id: str):
    job = next((j for j in jobs_db if j.get("job_id") == job_id), None)
    if not job:
        raise HTTPException(404, "Job not found")
    path = job.get("main_video_path") or job.get("output_video_path")
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Processed video not found on disk. Did you copy the storage folder?")
    return FileResponse(path, media_type="video/mp4", filename=f"statscout_{job_id}.mp4")

@app.get("/api/video/stream/{job_id}/{view}")
async def stream_video_view(job_id: str, view: str):
    view_map = {"main": "main_video_path", "pitch": "pitch_video_path", "voronoi": "voronoi_video_path"}
    field = view_map.get(view)
    if not field:
        raise HTTPException(400, f"Unknown view '{view}'.")
    
    job = next((j for j in jobs_db if j.get("job_id") == job_id), None)
    if not job:
        raise HTTPException(404, "Job not found")
    
    path = job.get(field)
    if not path and view == "main":
        path = job.get("output_video_path")
    if not path or not os.path.exists(path):
        raise HTTPException(404, f"Video for view '{view}' not found on disk. Did you copy the storage folder?")
    return FileResponse(path, media_type="video/mp4", filename=f"statscout_{job_id}_{view}.mp4")

@app.get("/api/video/demos", response_model=List[DemoSummary])
async def list_demos():
    demos = []
    demo_jobs = sorted([j for j in jobs_db if j.get("is_demo")], key=lambda x: x.get("demo_order", 0))
    for job in demo_jobs:
        urls = _build_video_urls(job, job.get("job_id"))
        demos.append(DemoSummary(
            job_id=job.get("job_id"),
            title=job.get("demo_title") or job.get("video_filename", "Demo clip"),
            description=job.get("demo_description"),
            output_video_url=urls.get("output_video_url") or f"/api/video/stream/{job.get('job_id')}",
            main_video_url=urls.get("main_video_url"),
            pitch_video_url=urls.get("pitch_video_url"),
            voronoi_video_url=urls.get("voronoi_video_url"),
        ))
    return demos

if __name__ == "__main__":
    import uvicorn
    print("=====================================================")
    print("🚀 STARTING STATSCOUT IN EXPO DEMO MODE")
    print("   AI models, Celery, and MongoDB are fully disabled.")
    print("   Serving cached results from seed_data.json.")
    print("=====================================================")
    uvicorn.run("demo_server:app", host="127.0.0.1", port=8000, reload=True)
