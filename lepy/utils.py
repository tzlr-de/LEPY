import os
import yaml
import re
import munch
import logging
import datetime as dt
import typing as T

from pathlib import Path
from skimage import transform as tr
from collections import defaultdict

from lepy.image import Image

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

def check_output(output: T.Optional[str], src: Path, *,
                 use_timestamp: bool = True,
                 force: bool = False) -> T.Optional[str]:
    if output is None:
        if use_timestamp:
            output = Path.cwd() / f"output/{dt.datetime.now():%Y-%m-%d_%H-%M-%S.%f}"
        else:
            src = Path(src)
            output = src.parent / f"{src.name}_result"

    logging.info(f"Outputs will be stored to {output}.")

    if Path(output).exists():
        logging.warning("Output folder already exists!")
        if force:
            logging.warning("Overwriting existing files.")
        elif input("Do you want save outputs into an existing folder? [y/N] ").lower() != "y":
            return None

    return output
