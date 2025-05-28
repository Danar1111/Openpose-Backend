import cv2
import numpy as np
import json
import os
import subprocess
import pandas as pd
from openpose import pyopenpose as op
from app.utils.image_converter import bytes_to_cv2
from app.services.model_predictor import predict_from_keypoints_df
from app.services.image_visualizer import generate_pose_visualization

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    v1 = a - b
    v2 = c - b
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def run_openpose(image, opWrapper):
    datum = op.Datum()
    datum.cvInputData = image
    datums = op.VectorDatum()
    datums.append(datum)
    opWrapper.emplaceAndPop(datums)
    return datum

def get_coords(keypoints, index):
    try:
        x, y, c = keypoints[0][index]
        return (int(x), int(y)) if c > 0.1 else None
    except:
        return None

def get_hand_coords(hand_keypoints, index):
    try:
        x, y, c = hand_keypoints[0][index]
        return (int(x), int(y)) if c > 0.1 else None
    except:
        return None

def detect_facing_direction(keypoints):
    score = 0
    nose = get_coords(keypoints, 0)
    left_shoulder = get_coords(keypoints, 5)
    left_elbow = get_coords(keypoints, 6)
    left_wrist = get_coords(keypoints, 7)
    left_knee = get_coords(keypoints, 13)

    if nose and left_shoulder:
        score += 1 if nose[0] < left_shoulder[0] else -1
    if left_shoulder and left_wrist:
        score += 1 if left_shoulder[0] > left_wrist[0] else -1
    if left_shoulder and left_knee:
        score += 1 if left_shoulder[0] > left_knee[0] else -1
    if left_shoulder and left_elbow:
        score += 1 if left_shoulder[0] > left_elbow[0] else -1

    return score

def process_pose_from_bytes(image_bytes):
    params = {
        "model_pose": "BODY_25",
        "hand": True,
        "number_people_max": 1,
        "net_resolution": "-1x160",
        "model_folder": "/root/openpose/models"
    }

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    image = bytes_to_cv2(image_bytes)
    datum = run_openpose(image, opWrapper)
    keypoints = datum.poseKeypoints

    processed_image = image_bytes
    processed_keypoints = keypoints
    is_flipped = False

    if keypoints is not None:
        direction_score = detect_facing_direction(keypoints)
        if direction_score > 0:
            print("Flip image ke kanan")
            flipped = cv2.flip(image, 1)
            datum = run_openpose(flipped, opWrapper)
            keypoints = datum.poseKeypoints
            is_flipped = True

    hand_kpts = datum.handKeypoints[1] if datum.handKeypoints is not None else None  # right hand

    # Ambil koordinat
    hip = get_coords(keypoints, 9)
    knee = get_coords(keypoints, 10)
    ankle = get_coords(keypoints, 11)
    shoulder = get_coords(keypoints, 2)
    elbow = get_coords(keypoints, 3)
    wrist = get_coords(keypoints, 4)
    neck = get_coords(keypoints, 1)
    head = get_coords(keypoints, 17)
    back = get_coords(keypoints, 8)

    thumb = get_hand_coords(hand_kpts, 4)
    index_finger = get_hand_coords(hand_kpts, 8)
    pinky = get_hand_coords(hand_kpts, 20)

    # Hitung sudut
    angles = {}

    if hip and knee and ankle:
        angles["sudut_lutut"] = calculate_angle(hip, knee, ankle) if hip and knee and ankle else 0
    if shoulder and elbow and wrist:
        angles["sudut_siku"] = calculate_angle(shoulder, elbow, wrist) if shoulder and elbow and wrist else 0
        angles["sudut_siku_rula"] = angles["sudut_siku"]
    if back and neck and head:
        angles["sudut_leher"] = calculate_angle(back, neck, head) if back and neck and head else 0
    if knee and hip and neck:
        angles["sudut_paha_punggung"] = calculate_angle(knee, hip, neck) if knee and hip and neck else 0
    if wrist and thumb and pinky:
        angles["sudut_pergelangan"] = calculate_angle(thumb, wrist, pinky) if thumb and wrist and pinky else 0
    if back and shoulder and elbow:
        angles["sudut_bahu"] = calculate_angle(back, shoulder, elbow) if back and shoulder and elbow else 0

    # Pastikan semua key sudut tersedia (isi 0 jika belum ada)
    required_keys = [
        "sudut_lutut",
        "sudut_siku",
        "sudut_siku_rula",
        "sudut_leher",
        "sudut_paha_punggung",
        "sudut_pergelangan",
        "sudut_bahu"
    ]

    for key in required_keys:
        if key not in angles:
            print(f"[DEBUG] Sudut '{key}' tidak tersedia, diset 0")
            angles[key] = 0

    print("[DEBUG] Final dict sudut:", angles)

    keypoint_df = get_keypoints(keypoints, hand_kpts)

    # Prediksi model
    hasil_prediksi = predict_from_keypoints_df(keypoint_df)

    # Generate visualisasi dan ambil path gambar
    gambar_path = generate_pose_visualization(processed_image, processed_keypoints, hasil_prediksi, is_flipped)

    # Tambahkan ke hasil prediksi
    hasil_prediksi["gambar_path"] = gambar_path

    return hasil_prediksi

def run_openpose_on_folder(image_folder, output_json_folder):
    openpose_bin_path = "/root/openpose/build/examples/openpose/openpose.bin"
    model_folder = "/root/openpose/models"

    command = [
        openpose_bin_path,
        "--image_dir", image_folder,
        "--write_json", output_json_folder,
        "--model_folder", model_folder,
        "--hand", "True",
        "--display", "0",
        "--render_pose", "0",
        "--disable_multi_thread"
    ]

    subprocess.run(command, check=True)

def process_openpose_results(json_folder, image_folder):
    results = []

    for file in sorted(os.listdir(json_folder)):
        if file.endswith(".json"):
            with open(os.path.join(json_folder, file)) as f:
                data = json.load(f)

            keypoints = None
            hand_right_keypoints = None

            if data['people']:
                person = data['people'][0]
                if "pose_keypoints_2d" in person and person["pose_keypoints_2d"]:
                    keypoints = np.array(person["pose_keypoints_2d"]).reshape(-1, 3)

                if "hand_right_keypoints_2d" in person and person["hand_right_keypoints_2d"]:
                    hand_right_keypoints = np.array(person["hand_right_keypoints_2d"]).reshape(-1, 3)

            if keypoints is not None:
                keypoint_df = get_keypoints(keypoints, hand_right_keypoints)

                hasil_prediksi = predict_from_keypoints_df(keypoint_df)

                # Path gambar input
                image_file = file.replace('.json', '.jpg')
                hasil_prediksi['gambar_path'] = os.path.join(image_folder, image_file)

                results.append(hasil_prediksi)

    return results

def get_keypoints(keypoints, hand_kpts):
    def get_point(index, keypoints_array):
        try:
            x, y, c = keypoints_array[index]
            return (x, y, c) if c > 0.01 else (0, 0, 0)
        except:
            return (0, 0, 0)
    
    keypoints_dict = {
        'hip': get_point(9, keypoints),
        'knee': get_point(10, keypoints),
        'ankle': get_point(11, keypoints),
        'shoulder': get_point(2, keypoints),
        'shoulder_left': get_point(5, keypoints),
        'elbow': get_point(3, keypoints),
        'elbow_left': get_point(6, keypoints),
        'wrist': get_point(4, keypoints),
        'hand_middle': get_point(9, hand_kpts) if hand_kpts is not None else (0, 0, 0),
        'back': get_point(8, keypoints),
        'neck': get_point(1, keypoints),
        'head': get_point(17, keypoints),
        'nose': get_point(0, keypoints)
    }

    print(keypoints_dict)

    return pd.DataFrame([keypoints_dict])

def check_video_direction(video_path, check_frame=10):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    direction_score = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == check_frame:
            # Deteksi arah
            params = {
                "model_pose": "BODY_25",
                "hand": True,
                "number_people_max": 1,
                "model_folder": "/root/openpose/models"
            }
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()

            datum = run_openpose(frame, opWrapper)
            keypoints = datum.poseKeypoints
            if keypoints is not None:
                direction_score = detect_facing_direction(keypoints)

            opWrapper.stop()
            break

        frame_idx += 1

    cap.release()
    return direction_score

def flip_video(video_path):
    cap = cv2.VideoCapture(video_path)

    flipped_path = video_path.replace(".mp4", "_flipped.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        flipped = cv2.flip(frame, 1)

        if out is None:
            height, width = flipped.shape[:2]
            out = cv2.VideoWriter(flipped_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

        out.write(flipped)

    cap.release()
    out.release()

    return flipped_path
