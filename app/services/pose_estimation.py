import cv2
import numpy as np
from openpose import pyopenpose as op
from app.utils.image_converter import bytes_to_cv2

def get_coords(kpts, index):
    try:
        x, y, c = kpts[0][index]
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

def run_openpose(image, opWrapper):
    datum = op.Datum()
    datum.cvInputData = image
    datums = op.VectorDatum()
    datums.append(datum)
    opWrapper.emplaceAndPop(datums)
    return datum

def safe_keypoints_to_list(kpts):
    return kpts.tolist() if kpts is not None else []

def process_pose_from_bytes(image_bytes):
    # Konfigurasi OpenPose
    params = {
        "model_pose": "BODY_25",
        "hand": True,
        "number_people_max": 1,
        "disable_multi_thread": False,
        "model_folder": "/root/openpose/models",
        "net_resolution": "656x368"
    }
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    image = bytes_to_cv2(image_bytes)

    # Proses awal
    datum = run_openpose(image, opWrapper)
    keypoints = datum.poseKeypoints

    if keypoints is not None:
        direction_score = detect_facing_direction(keypoints)
        print("Arah hadap:", "kiri" if direction_score > 0 else "kanan" if direction_score < 0 else "netral")

        if direction_score > 0:
            # Flip jika menghadap kiri
            print("Flip image ke kanan")
            flipped = cv2.flip(image, 1)
            datum = run_openpose(flipped, opWrapper)
            keypoints = datum.poseKeypoints

    # Return semua keypoints
    result = {
        "body": safe_keypoints_to_list(datum.poseKeypoints),
        "hand_left": safe_keypoints_to_list(datum.handKeypoints[0]) if datum.handKeypoints is not None else [],
        "hand_right": safe_keypoints_to_list(datum.handKeypoints[1]) if datum.handKeypoints is not None else [],
    }
    return result
