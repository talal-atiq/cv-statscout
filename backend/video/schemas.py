from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    DOWNLOADING_MODELS = "downloading_models"
    EXTRACTING_FRAMES = "extracting_frames"
    DETECTING = "detecting"
    CLASSIFYING_TEAMS = "classifying_teams"
    COMPUTING_METRICS = "computing_metrics"
    RENDERING = "rendering"
    DONE = "done"
    FAILED = "failed"


class VideoJob(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = 0
    status_message: str = ""
    created_at: datetime
    updated_at: datetime
    video_filename: str
    total_frames: int = 0
    processed_frames: int = 0
    output_video_path: Optional[str] = None
    analytics: Optional[dict] = None
    error: Optional[str] = None


class UploadResponse(BaseModel):
    job_id: str
    message: str


class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    status_message: str
    output_video_url: Optional[str] = None  # alias for main_video_url, kept for back-compat
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
