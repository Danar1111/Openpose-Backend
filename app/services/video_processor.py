import cv2
import numpy as np
import os
from datetime import datetime
from app.services.pose_estimation import process_pose_from_bytes
from app.services.job_manager import update_job
from app.utils.summarize_results import summarize_results

def process_video(video_bytes, job_id):
    # Simpan sementara videonya
    video_path = save_temp_video(video_bytes)

    direction_score = check_video_direction(video_path)

    if direction_score > 0:
        print("[DEBUG] Video menghadap kiri, flip dulu.")
        video_path = flip_video(video_path)

    folder_images = "temp_frames"
    json_output = "temp_json"

    sample_video_to_folder(video_path, folder_images)

    run_openpose_on_folder(folder_images, json_output)

    sampled_results = process_openpose_results(json_output, folder_images)

    os.remove(video_path)

    final_result = summarize_results(sampled_results)

    # Simpan hasil ke job
    update_job(job_id, final_result)

def save_temp_video(video_bytes):
    folder = "temp_videos"
    os.makedirs(folder, exist_ok=True)

    filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_video.mp4"
    path = os.path.join(folder, filename)

    with open(path, "wb") as f:
        f.write(video_bytes)

    return path

def sample_video_to_folder(video_path, output_folder, frame_interval=30):
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_files = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = f"{frame_count:06d}.jpg"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)
            saved_files.append(filepath)

        frame_count += 1

    cap.release()
    return saved_files
