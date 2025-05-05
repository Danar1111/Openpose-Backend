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
    required_columns = [
        "sudut_lutut",
        "sudut_siku",
        "sudut_siku_rula",
        "sudut_leher",
        "sudut_paha_punggung",
        "sudut_pergelangan",
        "sudut_bahu"
    ]
    
    data_input = {col: angle_dict.get(col, 0) for col in required_columns}
    df = pd.DataFrame([angle_dict])
    
    # Prediksi
    hasil_prediksi = model.predict_pose(df)
    
    # Konversi output ke native Python
    hasil_python = convert_to_python_type(hasil_prediksi)

    # Gabungkan hasil sudut ke hasil prediksi
    hasil_python["sudut"] = angle_dict
    return hasil_python
