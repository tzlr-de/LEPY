import os

from pathlib import Path
from skimage import transform as tr

def rescale(im, factor, **kwargs):
    if factor is None:
        return im
    dest_dtype = im.dtype
    return tr.rescale(im, factor, order=0, preserve_range=True, **kwargs).astype(dest_dtype)


def find_images(folder, *, ordered: bool = True, extensions=[".jpg", ".png"]):
    res = []
    for root, folders, fnames in os.walk(folder):
        for name in fnames:
            if Path(name).suffix.lower() not in extensions:
                continue
            res.append(Path(root) / name)

    if ordered:
        return sorted(res)
        
    return res