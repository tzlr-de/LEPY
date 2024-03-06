import cv2

def otsu(im):
    _, bin_im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_im
