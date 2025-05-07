from flask import Blueprint, request, jsonify, send_from_directory
from app.services.pose_estimation import process_pose_from_bytes
from app.services.video_processor import process_video
from app.services.job_manager import create_job, get_job
import os

pose_bp = Blueprint('pose', __name__)

@pose_bp.route('/predict/image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image_bytes = file.read()

    try:
        result = process_pose_from_bytes(image_bytes)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@pose_bp.route('/output_images/<path:filename>')
def serve_output_image(filename):
    directory = os.path.join(os.getcwd(), 'output_images')
    return send_from_directory(directory, filename, mimetype='image/png')

@pose_bp.route("/predict/video", methods=["POST"])
def predict_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['video']

    # Buat job
    job_id = create_job()

    job_folder = os.path.join("temp_jobs", job_id)
    os.makedirs(job_folder, exist_ok=True)

    video_path = os.path.join(job_folder, "video.mp4")
    file.save(video_path) 

    # Jalankan processing di background thread (biar nggak nungguin lama)
    import threading
    threading.Thread(target=process_video, args=(job_folder, job_id)).start()

    return jsonify({"job_id": job_id})

@pose_bp.route("/predict/video/result", methods=["GET"])
def get_video_result():
    job_id = request.args.get("job_id")

    if not job_id:
        return jsonify({"error": "Job ID is required"}), 400

    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify(job)