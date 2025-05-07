import cv2
import numpy as np
import os
import uuid
import shutil
from datetime import datetime
from app.services.pose_estimation import process_pose_from_bytes, check_video_direction, flip_video, run_openpose_on_folder, process_openpose_results
from app.services.job_manager import update_job
from app.utils.summarize_results import summarize_results

def process_video(video_bytes, job_id):
    job_folder, video_folder, frames_folder, json_output = create_job_folder()

    try:
        video_path = os.path.join(video_folder, "video.mp4")
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        direction_score = check_video_direction(video_path)

        if direction_score > 0:
            print("[DEBUG] Video menghadap kiri, flip dulu.")
            flipped_path = flip_video(video_path)
            video_path = flipped_path

        sample_video_to_folder(video_path, frames_folder)

        run_openpose_on_folder(frames_folder, json_output)

        sampled_results = process_openpose_results(json_output, frames_folder)

        final_result = summarize_results(sampled_results)

        update_job(job_id, final_result)

    finally:
        if os.path.exists(job_folder):
            shutil.rmtree(job_folder)
            print(f"[DEBUG] Folder job {job_folder} sudah dibersihkan.")

def create_job_folder():
    base_folder = "temp_jobs"
    os.makedirs(base_folder, exist_ok=True)

    job_id = str(uuid.uuid4())
    job_folder = os.path.join(base_folder, job_id)
    os.makedirs(job_folder, exist_ok=True)

    video_folder = job_folder
    frames_folder = os.path.join(job_folder, "frames")
    json_folder = os.path.join(job_folder, "json")

    os.makedirs(frames_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)

    return job_folder, video_folder, frames_folder, json_folder

def save_temp_video(video_bytes):
    folder = "temp_videos"
    os.makedirs(folder, exist_ok=True)

    filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_video.mp4"
    path = os.path.join(folder, filename)

    with open(path, "wb") as f:
        f.write(video_bytes)

    return path

def sample_video_to_folder(video_path, output_folder, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = f"{frame_count:06d}.jpg"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)

        frame_count += 1

    cap.release()
