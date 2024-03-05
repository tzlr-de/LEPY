import os
import yaml
import munch
import logging
import datetime as dt

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
            if Path(name).suffix.lower() not in extensions or name.startswith("."):
                continue
            res.append(Path(root) / name)

    if ordered:
        return sorted(res)
        
    return res

def read_config(path) -> munch.Munch:
    with open(path) as f:
        return munch.munchify(yaml.safe_load(f))
    
def check_output(args, *, use_timestamp: bool = True):
    folder = args.output
    if folder is None:
        if use_timestamp:
            folder = Path.cwd() / f"output/{dt.datetime.now():%Y-%m-%d_%H-%M-%S.%f}"
        else:
            src = Path(args.folder)
            folder = src.parent / f"{src.name}_result"

    if Path(folder).exists():
        logging.warning("Output folder already exists!")
        if input("Do you want save outputs into an existing folder? [y/N] ").lower() != "y":
            return None

    return folder