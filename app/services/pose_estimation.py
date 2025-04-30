import cv2
import numpy as np
from openpose import pyopenpose as op
from app.utils.image_converter import bytes_to_cv2

def process_pose_from_bytes(image_bytes):
    # Konfigurasi OpenPose
    params = {
        "model_pose": "BODY_25",
        "hand": False,  # sementara nonaktif, bisa diaktifkan nanti
        "number_people_max": 1,
        "disable_multi_thread": True,
        "model_folder": "/root/openpose/models"  # pastikan path ini sesuai lokasi model Anda
    }
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    image = bytes_to_cv2(image_bytes)

    datum = op.Datum()
    datum.cvInputData = image
    datum_ptr = op.Datum()
    datum_ptr.cvInputData = image

    datums = op.VectorDatum()
    datums.append(datum_ptr)

    opWrapper.emplaceAndPop(datums)


    # Misalnya outputnya adalah keypoints
    keypoints = datum.poseKeypoints.tolist() if datum.poseKeypoints is not None else []
    return {"keypoints": keypoints}
