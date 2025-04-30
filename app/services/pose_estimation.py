import cv2
import numpy as np
from openpose import pyopenpose as op
from app.utils.image_converter import bytes_to_cv2
from app.services.model_predictor import predict_from_angles

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
        "model_folder": "/root/openpose/models"
    }

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    image = bytes_to_cv2(image_bytes)
    datum = run_openpose(image, opWrapper)
    keypoints = datum.poseKeypoints

    if keypoints is not None:
        direction_score = detect_facing_direction(keypoints)
        if direction_score > 0:
            print("Flip image ke kanan")
            flipped = cv2.flip(image, 1)
            datum = run_openpose(flipped, opWrapper)
            keypoints = datum.poseKeypoints

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
        angles["sudut_lutut"] = calculate_angle(hip, knee, ankle) if hip and knee and ankle else None
    if shoulder and elbow and wrist:
        elbow_angle = calculate_angle(shoulder, elbow, wrist) if shoulder and elbow and wrist else None
        angles["sudut_siku"] = elbow_angle
        angles["sudut_siku_rula"] = elbow_angle
    if back and neck and head:
        angles["sudut_leher"] = calculate_angle(back, neck, head) if back and neck and head else None
    if knee and hip and neck:
        angles["sudut_paha_punggung"] = calculate_angle(knee, hip, neck) if knee and hip and neck else None
    if wrist and thumb and pinky:
        angles["sudut_pergelangan"] = calculate_angle(thumb, wrist, pinky) if thumb and wrist and pinky else None
    if back and shoulder and elbow:
        angles["sudut_bahu"] = calculate_angle(back, shoulder, elbow) if back and shoulder and elbow else None

    return predict_from_angles(angles)