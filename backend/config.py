from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "statscout_video"
    inference_server_url: str = "http://localhost:9001"
    roboflow_api_key: str = ""
    redis_url: str = "redis://localhost:6379/0"
    upload_dir: str = "./storage/uploads"
    processed_dir: str = "./storage/processed"
    frames_dir: str = "./storage/frames"
    frame_sample_rate: float = 1.0  # frames per second to process
    max_upload_size_mb: int = 2000

    class Config:
        env_file = ".env"


settings = Settings()
