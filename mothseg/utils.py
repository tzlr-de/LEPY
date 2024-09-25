import os
import yaml
import re
import munch
import logging
import datetime as dt

from pathlib import Path
from skimage import transform as tr
from collections import defaultdict

from mothseg.image import Image

def rescale(im, factor, **kwargs):
    if factor is None:
        return im
    dest_dtype = im.dtype
    return tr.rescale(im, factor, order=0, preserve_range=True, **kwargs).astype(dest_dtype)


def find_images(folder, *, config) -> dict:
    res = defaultdict(Image)

    rgb_regex = re.compile(config.rgb_regex)
    uv_regex = re.compile(config.uv_regex)
    for root, folders, fnames in os.walk(folder):
        for name in fnames:
            rgb_match = rgb_regex.match(name)
            uv_match = uv_regex.match(name)
            if uv_match:
                key = uv_match.group(1)
                curr_file: Image = res[key]
                if curr_file.key is None:
                    curr_file.key = key
                if curr_file.root is None:
                    curr_file.root = Path(root)
                curr_file.uv_fname = name

            elif rgb_match:
                key = rgb_match.group(1)
                curr_file: Image = res[key]
                if curr_file.key is None:
                    curr_file.key = key
                if curr_file.root is None:
                    curr_file.root = Path(root)
                curr_file.rgb_fname = name

            elif Path(name).suffix.lower() not in config.extensions or name.startswith("."):
                continue

            else:
                curr_file: Image = res[name]
                if curr_file.key is None:
                    curr_file.key = name
                if curr_file.root is None:
                    curr_file.root = Path(root)
                curr_file.rgb_fname = name


    if config.ordered:
        raise NotImplementedError("Ordered image loading is not implemented yet.")
        return sorted(res)

    return dict(res)

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

    logging.info(f"Outputs will be stored to {folder}.")

    if Path(folder).exists():
        logging.warning("Output folder already exists!")
        if input("Do you want save outputs into an existing folder? [y/N] ").lower() != "y":
            return None

    return folder
