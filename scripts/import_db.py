import json
import os
from bson import json_util
from pymongo import MongoClient

# Connect to the local MongoDB instance running in Docker
client = MongoClient("mongodb://localhost:27017")
db = client["statscout_video"]
collection = db["video_jobs"]

input_file = "seed_data.json"

if not os.path.exists(input_file):
    print(f"Error: Could not find {input_file}. Make sure it is in the same directory.")
    exit(1)

print(f"Loading records from {input_file}...")
with open(input_file, "r", encoding="utf-8") as f:
    jobs = json_util.loads(f.read())

if not jobs:
    print("No records found in the JSON file.")
    exit(0)

# Import records, ignoring duplicates based on job_id
inserted = 0
for job in jobs:
    try:
        # Use update_one with upsert so we don't crash on existing records
        collection.update_one({"job_id": job["job_id"]}, {"$set": job}, upsert=True)
        inserted += 1
    except Exception as e:
        print(f"Failed to insert job {job.get('job_id')}: {e}")

print(f"Successfully imported/updated {inserted} records in the database.")
