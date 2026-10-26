import os
import pickle

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SVM_PATH = os.path.join(BASE_DIR, "models", "svm_model.pkl")
LE_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

CONFIDENCE_THRESHOLD = 0.55

clasificator_svm = None
encoder_etichete = None


def _load_models():
    global clasificator_svm, encoder_etichete
    if clasificator_svm is not None:
        return clasificator_svm, encoder_etichete

    if not os.path.exists(SVM_PATH):
        raise FileNotFoundError(f"svm_model.pkl negasit la: {SVM_PATH}")
    if not os.path.exists(LE_PATH):
        raise FileNotFoundError(f"label_encoder.pkl negasit la: {LE_PATH}")

    with open(SVM_PATH, "rb") as f:
        clasificator_svm = pickle.load(f)
    with open(LE_PATH, "rb") as f:
        encoder_etichete = pickle.load(f)

    print(f"Modele incarcate: {len(encoder_etichete.classes_)} clase")
    return clasificator_svm, encoder_etichete


def predict(embedding: np.ndarray, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    clasificator_svm, encoder_etichete = _load_models()

    embedding_2d = embedding.reshape(1, -1)
    probabilitati = clasificator_svm.predict_proba(embedding_2d)[0]
    idx_maxim = int(np.argmax(probabilitati))
    incredere = float(probabilitati[idx_maxim])
    if incredere >= threshold:
        nume = encoder_etichete.classes_[idx_maxim]
    else:
        nume = "Necunoscut"

    if incredere >= threshold:
        este_cunoscut = True
    else:
        este_cunoscut = False

    rezultat = {
        "name": nume,
        "confidence": incredere,
        "known": este_cunoscut,
    }

    return rezultat


def get_known_classes() -> list[str]:
    _, encoder_etichete = _load_models()
    return list(encoder_etichete.classes_)
