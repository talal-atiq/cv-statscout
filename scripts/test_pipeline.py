"""
Standalone test script. Run from the project root:
    python scripts/test_pipeline.py --video path/to/your/test_clip.mp4

Tests the full CV pipeline without any web server or Celery.
Output video is written to ./storage/processed/test_output.mp4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from backend.video.processor import process_video


def main():
    parser = argparse.ArgumentParser(description="StatScout pipeline smoke test")
    parser.add_argument("--video", required=True, help="Path to input video file")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: video file not found: {args.video}")
        sys.exit(1)

    def progress(status, pct, msg):
        print(f"[{pct:3d}%] [{status}] {msg}")

    print(f"Starting pipeline on: {args.video}")
    result = process_video(
        job_id="test_run",
        video_path=args.video,
        progress_callback=progress,
    )

    print("\n✅ Pipeline complete!")
    print(f"Output video : {result['output_video_path']}")
    print(f"Possession   : {result['analytics']['possession']}")
    print(f"Players tracked: {len(result['analytics']['players'])}")


if __name__ == "__main__":
    main()
