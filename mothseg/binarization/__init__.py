from mothseg.binarization.otsu import otsu
from mothseg.binarization.gauss_local import gauss_local
from mothseg.binarization.grabcut import stable_grabcut
from mothseg.binarization.grabcut import grabcut

def binarize(im, chan, method: str):

    if method == "otsu":
        return otsu(chan)
        
    elif method == "gauss_local":
        return gauss_local(chan)
        
    elif method.startswith("grabcut"):
        return stable_grabcut(im, chan, n_rounds=5, 
                              plus="+" in method,
                              use_otsu="otsu" in method,
                              rescale_factor=1/4)

    else:
        raise ValueError(f"Unknown method: {method}!")

__all__ = [
    "binarize",
    "otsu",
    "gauss_local",
    "grabcut",
    "stable_grabcut"
]