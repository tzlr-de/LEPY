
import numpy as np

from PIL import Image as PILImage
from backgroundremover.u2net.detect import load_model
from backgroundremover.u2net.detect import predict

def unet(im: np.ndarray, *, model_name: str = "u2net") -> np.ndarray:
    global MODEL
    if MODEL is None:
        MODEL = load_model(model_name=model_name)
    mask = predict(MODEL, im).convert("L")
    mask = mask.resize(im.shape[:2][::-1], PILImage.LANCZOS)
    return np.where(np.array(mask) < 128, 0, 1).astype(np.uint8)

MODEL = None
