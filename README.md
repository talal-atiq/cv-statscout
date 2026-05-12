# StatScout Video Analysis Module

A standalone football match video analysis microservice. Upload a match video, get back:

- Annotated video with bounding boxes, player IDs, team colours, and speed labels
- 2D top-down pitch map baked into the output video
- JSON analytics: possession %, per-player speed timeline, distance covered, heatmap data

All CV inference runs **locally** via the Roboflow `inference` local server — no per-call cloud costs.

---

## Quick Start

### 1. Install the `sports` package (not on PyPI)

```bash
pip install git+https://github.com/roboflow/sports.git
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Start MongoDB + Redis

```bash
docker compose up -d
```

### 4. Start the Roboflow local inference server

```bash
inference server start
# Runs at http://localhost:9001
# Downloads model weights on first inference call (~200 MB total)
```

### 5. Configure environment

```bash
cp .env.example .env
# Edit .env and set ROBOFLOW_API_KEY
```

### 6. Test the pipeline standalone

```bash
python scripts/test_pipeline.py --video /path/to/football_clip.mp4
```

### 7. Start the Celery worker

```bash
celery -A backend.celery_app.celery_app worker --loglevel=info
```

### 8. Start the API server

```bash
uvicorn backend.main:app --reload --port 8000
```

### 9. Start the frontend

```bash
cd frontend
npm install
npm run dev
# http://localhost:5173
```

---

## Models Used

| Model | Roboflow ID | Purpose |
|---|---|---|
| Player/Ball/Referee Detection | `football-players-detection-3zvbc/1` | YOLOv8 — detects all pitch entities |
| Pitch Keypoint Detection | `football-field-detection-f07vi/14` | 32 pitch landmarks for homography |

---

## Architecture

```
Upload → FastAPI → Celery Task → process_video()
                                    ├── InferenceHTTPClient (local)
                                    ├── sv.ByteTrack
                                    ├── compute_homography()
                                    ├── TeamClassifier (SigLIP + KMeans)
                                    ├── build_analytics()
                                    └── draw_frame() + draw_pitch_map()
```

See `STATSCOUT_VIDEO_MODULE.md` for the full implementation specification.
