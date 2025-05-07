import cv2
import os
import shutil
from app.services.pose_estimation import check_video_direction, flip_video, run_openpose_on_folder, process_openpose_results
from app.services.job_manager import update_job
from app.utils.summarize_results import summarize_results

def process_video(job_folder, job_id):
    video_path = os.path.join(job_folder, "video.mp4")

    frames_folder = os.path.join(job_folder, "frames")
    json_folder = os.path.join(job_folder, "json")

    os.makedirs(frames_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)

    try:
        direction_score = check_video_direction(video_path)

        if direction_score > 0:
            video_path = flip_video(video_path)

        sample_video_to_folder(video_path, frames_folder)

        run_openpose_on_folder(frames_folder, json_folder)

        sampled_results = process_openpose_results(json_folder, frames_folder)

        final_result = summarize_results(sampled_results)

        update_job(job_id, final_result)

    finally:
        shutil.rmtree(job_folder)
        print(f"[DEBUG] Folder job {job_folder} sudah dibersihkan.")

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
