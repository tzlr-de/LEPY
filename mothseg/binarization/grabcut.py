import cv2
import numpy as np

from mothseg.utils import rescale


def stable_grabcut(im, chan, *, n_rounds: int, 
                   plus: bool,
                   use_otsu: bool,
                   rescale_factor: float = 1/4,
                   iterations: int = 5):
    
    _im = rescale(im, factor=rescale_factor, channel_axis=2).astype(np.uint8)

    bin_im = np.zeros(_im.shape[:2], dtype=np.float32)
    for i in range(n_rounds):
        bin_im += grabcut(_im, 
                          iterations=iterations,
                          plus=plus,
                          use_otsu=use_otsu,
                          chan=rescale(chan, factor=rescale_factor) if use_otsu else None
                        ) 

    return rescale((bin_im / n_rounds).astype(np.uint8), factor=1/rescale_factor)

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
