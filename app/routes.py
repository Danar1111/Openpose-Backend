from flask import Blueprint, request, jsonify, send_from_directory
from app.services.pose_estimation import process_pose_from_bytes
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