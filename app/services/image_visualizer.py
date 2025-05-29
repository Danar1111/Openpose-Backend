import os
import cv2
import numpy as np
from datetime import datetime

def get_color_by_score(score):
    if score == 0:
        return (144, 238, 144)  # Light green
    elif score == 1:
        return (0, 255, 0)      # Green
    elif score == 2:
        return (173, 255, 47)   # Yellow-green
    elif score == 3:
        return (0, 255, 255)    # Yellow
    elif score == 4:
        return (0, 165, 255)    # Orange
    elif score == 5:
        return (0, 140, 255)    # Darker Orange
    elif score == 6:
        return (0, 69, 255)     # Orange-red
    else:  # score >= 7
        return (0, 0, 255)      # Red

def get_risk_label(score):
    if score <= 1:
        return "Negligible"
    elif score <= 3:
        return "Low"
    elif score <= 7:
        return "Medium"
    elif score <= 10:
        return "High"
    else:
        return "Very High"

def generate_pose_visualization(image_bytes, keypoints, hasil_prediksi, is_flipped):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    mapping = {
        "sudut_lutut": hasil_prediksi.get("reba_leg_score", 0),
        "sudut_siku": hasil_prediksi.get("rula_lower_arm_score", 0),
        "sudut_leher": hasil_prediksi.get("rula_neck_score", 0),
        "sudut_punggung": hasil_prediksi.get("reba_trunk_score", 0),
        "sudut_pergelangan": hasil_prediksi.get("rula_wrist_score", 0),
        "sudut_bahu": hasil_prediksi.get("rula_upper_arm_score", 0),
    }

    point_mapping = {
        "sudut_lutut": 13 if is_flipped else 10,
        "sudut_siku": 6 if is_flipped else 3,
        "sudut_leher": 18 if is_flipped else 17,
        "sudut_punggung": 12 if is_flipped else 9,
        "sudut_pergelangan": 7 if is_flipped else 4,
        "sudut_bahu": 5 if is_flipped else 2,
    }

    for key, index in point_mapping.items():
        try:
            x, y, conf = keypoints[0][index]
            if conf > 0.1:
                color = get_color_by_score(mapping[key])

                overlay = img.copy()

                cv2.circle(overlay, (int(x), int(y)), 25, color, -1)

                alpha = 0.4
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                sudut_val = hasil_prediksi.get("details", {}).get(key, None)
                if sudut_val is not None:
                    label = f"{sudut_val:.1f}"

                    text_x = int(x) + 30
                    text_y = int(y) - 10

                    # Gambar outline putih (lebih tebal)
                    cv2.putText(img, label, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 4)

                    # Gambar teks utama hitam (di atas outline)
                    cv2.putText(img, label, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        except:
            continue
    
    # === Skor & label
    rula_final = hasil_prediksi.get("rula_final_score", 0)
    reba_final = hasil_prediksi.get("reba_final_score", 0)
    rula_label = get_risk_label(rula_final)
    reba_label = get_risk_label(reba_final)

    def label_color(label):
        return {
            "Negligible": (144, 238, 144),
            "Low": (0, 255, 0),
            "Medium": (0, 255, 255),
            "High": (0, 165, 255),
            "Very High": (0, 0, 255)
        }.get(label, (255, 255, 255))

    # === Kotak kecil di kanan atas (lebih kecil dari sebelumnya)
    badge_x = 10
    badge_y = 10
    badge_w = 150
    badge_h = 40

    cv2.rectangle(img, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (50, 50, 50), -1)

    # === Tulisan kecil, tapi tetap jelas
    font_scale = 0.5
    font_thickness = 1

    cv2.putText(img, f"RULA: {rula_final}", (badge_x + 10, badge_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color(rula_label), font_thickness)
    cv2.putText(img, f"REBA: {reba_final}", (badge_x + 10, badge_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color(reba_label), font_thickness)

    folder_path = os.path.join("output_images", datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(folder_path, exist_ok=True)

    filename = datetime.now().strftime("%H%M%S_%f") + "_hasil.png"
    filepath = os.path.join(folder_path, filename)
    link_image = "https://vps.danar.site/" + filepath
    cv2.imwrite(filepath, img)

    return link_image
