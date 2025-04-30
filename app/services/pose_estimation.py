import cv2
import numpy as np
from openpose import pyopenpose as op
from app.utils.image_converter import bytes_to_cv2

def process_pose_from_bytes(image_bytes):
    # Konfigurasi OpenPose
    params = {
        "model_folder": "/root/openpose/models/",
        "net_resolution": "-1x368",
        "disable_multi_thread": True,
    }
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    
    image = bytes_to_cv2(image_bytes)

    datum = op.Datum()
    datum.cvInputData = image
    opWrapper.emplaceAndPop([datum])

    # Misalnya outputnya adalah keypoints
    keypoints = datum.poseKeypoints.tolist() if datum.poseKeypoints is not None else []
    return {"keypoints": keypoints}
