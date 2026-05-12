import json
import os
from bson import json_util
from pymongo import MongoClient

# Connect to the local MongoDB instance running in Docker
client = MongoClient("mongodb://localhost:27017")
db = client["statscout_video"]
collection = db["video_jobs"]

# Fetch all jobs
print("Fetching records from database...")
jobs = list(collection.find({}))

# Export to a JSON file
output_file = "seed_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(json_util.dumps(jobs, indent=4))

print(f"Successfully exported {len(jobs)} records to {output_file}")
print("You can now commit this JSON file to GitHub.")
