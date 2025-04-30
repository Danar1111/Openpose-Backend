from openpose import pyopenpose as op
import cv2

def process_image_with_openpose(image):
    params = {
        "model_folder": "/root/openpose/models",
        "model_pose": "BODY_25",
        "hand": True,
        "number_people_max": 1,
        "disable_multi_thread": True,
        "num_gpu": 0,
        "num_gpu_start": 0
    }

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    datum = op.Datum()
    datum.cvInputData = image

    data = op.VectorDatum()
    data.append(datum)
    opWrapper.emplaceAndPop(data)

    keypoints = datum.poseKeypoints[0] if datum.poseKeypoints is not None else []

    hand_keypoints = []
    if datum.handKeypoints and isinstance(datum.handKeypoints[1], list) and len(datum.handKeypoints[1]) > 0:
        if datum.handKeypoints[1][0] is not None:
            hand_keypoints = datum.handKeypoints[1][0]

    return keypoints, hand_keypoints
