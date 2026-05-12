from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.video.routes import router as video_router
from backend.database import ensure_indexes

app = FastAPI(title="StatScout Video Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video_router)


@app.on_event("startup")
async def on_startup():
    await ensure_indexes()


@app.get("/health")
def health():
    return {"status": "ok"}
