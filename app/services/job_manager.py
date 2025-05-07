import uuid
import time

jobs = {}

JOB_EXPIRE_SECONDS = 86400  # 1 hari

def generate_random_id():
    return str(uuid.uuid4())

def create_job():
    job_id = generate_random_id()
    jobs[job_id] = {
        "status": "processing",
        "result": None,
        "created_at": time.time()
    }
    return job_id

def update_job(job_id, result):
    if job_id in jobs:
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result
        jobs[job_id]["finished_at"] = time.time()
        jobs[job_id]["expire_at"] = time.time() + JOB_EXPIRE_SECONDS

def get_job(job_id):
    job = jobs.get(job_id)

    # Cek apakah job sudah expired
    if job and "expire_at" in job:
        if time.time() > job["expire_at"]:
            del jobs[job_id]
            return None

    return job
