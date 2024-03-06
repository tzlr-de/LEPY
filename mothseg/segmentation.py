import cv2
import typing as T
import numpy as np
import scipy as sp

from mothseg.binarization import binarize


def segment(im, *, method: str = "grabcut+otsu", 
            channel: str = "saturation", 
            ksize: T.Optional[int] = None, 
            fill_holes: bool = True):
    
    hsv_im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    height, width, *_ = hsv_im.shape
    H, S, V = hsv_im.transpose(2, 0, 1)

    chan = {
        "hue": H,
        "saturation": S,
        "intensity": V,
        "value": V,
        "gray": V,
        "grey": V,
    }.get(channel)

    assert chan is not None, \
        f"Could not select desired channel: {channel}"
    
    if ksize is not None:
        chan = cv2.GaussianBlur(chan, (ksize, ksize), 0)
    
    bin_im = binarize(im, chan, method=method)

    contours, _ = cv2.findContours(bin_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    largest_contour = contours[0]

    stats = {
        
        'median-intensity': float(np.median(V)),
        'mean-intensity': float(np.mean(V)),
        'stddev-intensity': float(np.std(V)),
        'median-saturation': float(np.median(S)),
        'mean-saturation': float(np.mean(S)),
        'stddev-saturation': float(np.std(S)),
        'median-hue': float(np.median(H)),
        'mean-hue': float(np.mean(H)),
        'stddev-hue': float(np.std(H)),
        'image-width': width,
        'image-height': height,
        'seg-absolute-size': len(V),
        'seg-relative-size': len(V) / float( hsv_im.shape[0] * hsv_im.shape[1] ),
        
        'c-length': len(largest_contour),
        'c-area': cv2.contourArea(largest_contour),

        # compute bounding box
        'c-xmin': int(np.amin( largest_contour[:, 0, 0] )),
        'c-xmax': int(np.amax( largest_contour[:, 0, 0] )),
        'c-ymin': int(np.amin( largest_contour[:, 0, 1] )),
        'c-ymax': int(np.amax( largest_contour[:, 0, 1] )),

    }

    if fill_holes:
        bin_im = sp.ndimage.binary_fill_holes(bin_im).astype(bin_im.dtype)

    return chan, stats, largest_contour[:, 0], bin_im

