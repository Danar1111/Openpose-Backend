import cloudpickle
import pandas as pd
import json
import os

# Load model saat modul diimpor
MODEL_PATH = os.path.abspath("./app/models/wrapped_ergo_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = cloudpickle.load(f)

def convert_to_python_type(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_type(i) for i in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj

def predict_from_angles(angle_dict):
    # Susun DataFrame dari sudut
    df = pd.DataFrame([angle_dict])
    
    # Prediksi
    hasil_prediksi = model.predict_pose(df)
    
    # Konversi output ke native Python
    hasil_python = convert_to_python_type(hasil_prediksi)
    return hasil_python
