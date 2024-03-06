import cv2

def gauss_local(im, *, block_size: int = 31, C: int = 0, max_value: int = 255):
    return cv2.adaptiveThreshold(im,
            maxValue=max_value,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=C)
