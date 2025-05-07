import uuid

jobs = {}

def generate_random_id():
    return str(uuid.uuid4())

def create_job():
    job_id = generate_random_id()
    jobs[job_id] = {"status": "processing", "result": None}
    return job_id

def update_job(job_id, result):
    jobs[job_id]["status"] = "done"
    jobs[job_id]["result"] = result

def get_job(job_id):
    return jobs.get(job_id, None)
