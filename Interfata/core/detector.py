import math
import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MP_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
MP_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

_detector = None


def _get_detector():
    global _detector
    if _detector is not None:
        return _detector

    if not os.path.exists(MP_MODEL_PATH):
        os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
        urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_PATH)

    base_options = python.BaseOptions(model_asset_path=MP_MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.3,
    )
    _detector = vision.FaceLandmarker.create_from_options(options)
    return _detector


def align_face(img_bgr: np.ndarray) -> np.ndarray | None:
    detector = _get_detector()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    rezultat_detectie = detector.detect(mp_img)

    if not rezultat_detectie.face_landmarks:
        return None

    puncte = rezultat_detectie.face_landmarks[0]
    inaltime, latime = img_bgr.shape[:2]

    coordonate_x = []
    coordonate_y = []
    for punct in puncte:
        coordonate_x.append(punct.x * latime)
        coordonate_y.append(punct.y * inaltime)

    x_min = int(max(0, min(coordonate_x) - 20))
    x_max = int(min(latime, max(coordonate_x) + 20))
    y_min = int(max(0, min(coordonate_y) - 20))
    y_max = int(min(inaltime, max(coordonate_y) + 20))
    fata = img_bgr[y_min:y_max, x_min:x_max]

    if fata.size == 0:
        return None

    ochi_stang = (
        int(puncte[33].x * latime) - x_min,
        int(puncte[33].y * inaltime) - y_min,
    )
    ochi_drept = (
        int(puncte[263].x * latime) - x_min,
        int(puncte[263].y * inaltime) - y_min,
    )
    delta_x = ochi_drept[0] - ochi_stang[0]
    delta_y = ochi_drept[1] - ochi_stang[1]
    unghi_rotire = math.degrees(math.atan2(delta_y, delta_x))

    h_fata, w_fata = fata.shape[:2]
    matrice_rotire = cv2.getRotationMatrix2D((w_fata // 2, h_fata // 2), unghi_rotire, 1.0)
    fata_rotita = cv2.warpAffine(fata, matrice_rotire, (w_fata, h_fata))

    fata_finala = cv2.resize(fata_rotita, (160, 160))
    fata_rgb = cv2.cvtColor(fata_finala, cv2.COLOR_BGR2RGB)
    return fata_rgb
