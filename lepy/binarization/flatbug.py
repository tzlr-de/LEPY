import numpy as np

from warnings import warn
# from matplotlib import pyplot as plt

try:
    import torch as th
    from flat_bug.predictor import Predictor
    from flat_bug.config import DEFAULT_CFG
    from flat_bug.geometric import contours_to_masks
    CFG = dict(DEFAULT_CFG)
    CFG["MIN_MAX_OBJ_SIZE"] = (300**2, int(10e8))
except ImportError:
    Predictor = None


def flatbug(im: np.ndarray, *,
            device: str = "cuda:0", dtype: str = "float16") -> np.ndarray:
    global MODEL, CFG
    if MODEL is None:
        assert Predictor is not None, "flat_bug package is not installed!"
        if not th.cuda.is_available():
            warn("Using CPU for flatbug model, performance may be slow!")
            device, dtype = "cpu", "float32"
        MODEL = Predictor(device=device, dtype=dtype, cfg=CFG)
    arr = th.as_tensor(im, device=MODEL._device).permute(2, 0, 1)
    prediction = MODEL(arr)
    if len(prediction.confs) == 0:
        warn("No predictions found, returning empty mask.")
        # plt.figure(figsize=(16, 10))
        # plt.imshow(im)
        # plt.savefig("empty_pred.png")
        # plt.close()
        # breakpoint()
        return np.zeros(im.shape[:2], dtype=np.uint8)

    conf = prediction.confs[0]
    if conf < 0.5:
        warn(f"Confidence {conf:.2f} is below threshold, returning empty mask.")
        return np.zeros(im.shape[:2], dtype=np.uint8)

    contour = prediction.contours[0]
    mask = contours_to_masks([contour.round().long()], *im.shape[:2])
    return mask[0].cpu().numpy().astype(np.uint8)


    # for k, v in prediction.json_data.items():
    #     v = str(v)
    #     if len(v) > 100:
    #         v = v[:100] + "..."
    #     print(f"{k}: {v}")

    # img = prediction.plot(
    #     masks=False, boxes=True, confidence=False
    # )
    # plt.figure(figsize=(16, 10))
    # plt.imshow(img)
    # plt.savefig("flatbug_prediction.png")
    # plt.close()
    # contours = prediction.contours
    # print([c.shape for c in contours])
    # breakpoint()
    # masks = contours_to_masks([c.round().long()for c in contours], *im.shape[:2])

    # mask, conf = prediction.crop_masks[0], prediction.confs[0]
    # if conf < 0.5:
    #     warn(f"Confidence {conf:.2f} is below threshold, returning empty mask.")

    # return mask.cpu().numpy().astype(np.uint8)
    # return np.where(np.array(mask) < 128, 0, 1).astype(np.uint8)

MODEL = None


__ALL__ = [
    "flatbug",
]
