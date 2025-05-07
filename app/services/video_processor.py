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

    # Buka video dengan OpenCV
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sampled_results = []
    frame_interval = 30  # Ambil setiap 30 frame (sekitar 1 detik kalau 30fps)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Ambil hanya frame sesuai interval
        if frame_count % frame_interval == 0:
            # Encode frame ke bytes
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Proses frame seperti proses foto
            result = process_pose_from_bytes(frame_bytes)

            sampled_results.append(result)

        frame_count += 1

    cap.release()
    os.remove(video_path)

    # Hitung hasil akhir dari semua frame
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