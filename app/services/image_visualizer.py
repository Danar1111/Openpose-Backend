import os
import cv2
import numpy as np
from datetime import datetime

def get_color_by_score(score):
    if score == 0:
        return (0, 255, 0)  # Hijau
    elif score == 1:
        return (0, 255, 255)  # Kuning
    else:
        return (0, 0, 255)  # Merah

def generate_pose_visualization(image_bytes, keypoints, hasil_prediksi):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    mapping = {
        "sudut_lutut": hasil_prediksi.get("skor_lutut_reba", 0),
        "sudut_siku": hasil_prediksi.get("skor_siku_reba", 0),
        "sudut_leher": hasil_prediksi.get("skor_leher_reba", 0),
        "sudut_paha_punggung": hasil_prediksi.get("skor_trunk_reba", 0),
        "sudut_pergelangan": hasil_prediksi.get("skor_pergelangan_rula", 0),
        "sudut_bahu": hasil_prediksi.get("skor_bahu_rula", 0),
    }

    point_mapping = {
        "sudut_lutut": 10,
        "sudut_siku": 3,
        "sudut_leher": 1,
        "sudut_paha_punggung": 9,
        "sudut_pergelangan": 4,
        "sudut_bahu": 2,
    }

    for key, index in point_mapping.items():
        try:
            x, y, conf = keypoints[0][index]
            if conf > 0.1:
                color = get_color_by_score(mapping[key])
                overlay = img.copy()
                cv2.circle(overlay, (int(x), int(y)), 20, color, -1)
                img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        except:
            continue

    folder_path = os.path.join("output_images", datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(folder_path, exist_ok=True)

    filename = datetime.now().strftime("%H%M%S") + "_hasil.png"
    filepath = os.path.join(folder_path, filename)
    cv2.imwrite(filepath, img)

    return filepath
