import cv2
import typing as T
import numpy as np
import scipy as sp

from mothseg.utils import rescale

def segment(im, *, method: str = "otsu", channel: str = "saturation", ksize: T.Optional[int] = None):
    hsv_im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
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
    
    thresh = None
    if method == "otsu":
        thresh, bin_im = cv2.threshold(chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    elif method == "gauss_local":
        bin_im = cv2.adaptiveThreshold(chan,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=31,
            C=0)
        
    elif method.startswith("grabcut"):
        n_rounds = 5
        factor = 0.25
        _im = rescale(im, factor=factor, channel_axis=2).astype(np.uint8)
        bin_im = np.zeros(_im.shape[:2], dtype=np.float32)
        for i in range(n_rounds):
            bin_im += grabcut(_im, 
                             iterations=5,
                             plus="+" in method,
                             use_otsu="otsu" in method,
                             chan=rescale(chan, factor=factor) if "otsu" in method else None
                            ) 

        bin_im = rescale((bin_im / n_rounds).astype(np.uint8), factor=1/factor)
    else:
        raise ValueError(f"Unknown method: {method}!")

    contours, hierarchy = cv2.findContours(bin_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    largest_contour = contours[0]

    stats = {
        
        'median-intensity': np.median(V),
        'mean-intensity': np.mean(V),
        'stddev-intensity': np.std(V),
        'median-saturation': np.median(S),
        'mean-saturation': np.mean(S),
        'stddev-saturation': np.std(S),
        'median-hue': np.median(H),
        'mean-hue': np.mean(H),
        'stddev-hue': np.std(H),
        'seg-absolute-size': len(V),
        'seg-relative-size': len(V) / float( hsv_im.shape[0] * hsv_im.shape[1] ),
        
        'c-length': len(largest_contour),
        'c-area': cv2.contourArea(largest_contour),

        # compute bounding box
        'c-xmin': np.amin( largest_contour[:, 0, 0] ),
        'c-xmax': np.amax( largest_contour[:, 0, 0] ),
        'c-ymin': np.amin( largest_contour[:, 0, 1] ),
        'c-ymax': np.amax( largest_contour[:, 0, 1] ),

    }

    return chan, stats, largest_contour[:, 0], sp.ndimage.binary_fill_holes(bin_im).astype(bin_im.dtype)


def grabcut(im, *, iterations: int = 5, plus: bool = False, use_otsu: bool = False, chan = None):
    h, w, *c = im.shape
    grabcut_mask = np.zeros(shape=im.shape[:2], dtype=np.uint8)
    grabcut_rect = (int(w * 0.05), int(h * 0.1), 
                    int(w * 0.9), int(h * 0.8))

    cv2.grabCut(im, 
                grabcut_mask, grabcut_rect, 
                np.zeros((1,65), np.float64), np.zeros((1,65), np.float64),
                iterations, cv2.GC_INIT_WITH_RECT )
    
    if plus:
        center = (w//2, h//2)
        
        rects = [
            ((center, (int(w * 0.05), int(h * 0.3)), 0), cv2.GC_PR_FGD),
            ((center, (int(w * 0.5), int(h * 0.03)), 0), cv2.GC_PR_FGD),
            
            (((0, 0), (int(w * 0.5), int(h * 0.5)), 0), cv2.GC_BGD),
            (((0, h), (int(w * 0.5), int(h * 0.5)), 0), cv2.GC_BGD),
            (((w, 0), (int(w * 0.5), int(h * 0.5)), 0), cv2.GC_BGD),
            (((w, h), (int(w * 0.5), int(h * 0.5)), 0), cv2.GC_BGD),
        ]
        
        for rect, lab in rects:
            cv2.ellipse(grabcut_mask, rect, lab, -1)
        
        if use_otsu and chan is not None:
            _, bin_im = cv2.threshold(chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            grabcut_mask[bin_im == 255] = cv2.GC_FGD
        
        cv2.grabCut(im, 
                    grabcut_mask, None, 
                    np.zeros((1,65), np.float64), np.zeros((1,65), np.float64),
                    iterations, cv2.GC_INIT_WITH_MASK )
        
    return np.where((grabcut_mask==cv2.GC_BGD) | (grabcut_mask==cv2.GC_PR_BGD), 0, 1).astype(np.uint8)
