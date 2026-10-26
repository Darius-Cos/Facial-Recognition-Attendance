import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

facenet_model = None
device_name = None


def _get_model():
    global facenet_model, device_name
    if facenet_model is not None:
        return facenet_model, device_name

    if torch.cuda.is_available():
        device_name = "cuda"
    else:
        device_name = "cpu"
    facenet_model = InceptionResnetV1(pretrained="vggface2").eval().to(device_name)
    print(f"FaceNet incarcat pe {device_name}")
    return facenet_model, device_name


def get_embedding(img_rgb_160: np.ndarray) -> np.ndarray:
    if img_rgb_160.shape!= (160, 160, 3):
        raise ValueError(f"imagine invalida: {img_rgb_160.shape}, asteptat (160, 160, 3)")

    model_facenet, device = _get_model()

    imagine_norm= (img_rgb_160.astype(np.float32) - 127.5) / 128.0
    tensor_imagine= torch.tensor(imagine_norm)
    tensor_imagine= tensor_imagine.permute(2, 0, 1)
    tensor_imagine= tensor_imagine.unsqueeze(0)
    tensor_imagine= tensor_imagine.to(device)

    with torch.no_grad():
        vector = model_facenet(tensor_imagine)

    return vector.cpu().numpy()[0]
