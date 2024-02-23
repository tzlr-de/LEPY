#!/usr/bin/env python
from __future__ import annotations

if __name__ != '__main__':
    raise Exception("Do not import me!")

import cv2
import numpy as np
import scalebar
import logging


from matplotlib.pyplot import imread
from tqdm.auto import tqdm

import mothseg

from mothseg import visualization as vis
from mothseg import parser
from mothseg import utils


def main(args):
    images = utils.find_images(args.folder)

    logging.info(f"Found {len(images):,d} images in {args.folder}")
    first10 = '\n'.join(map(str, images[:10]))
    logging.debug(f"Here are the first 10: \n{first10}")

    if not args.yes and input("Do you want to proceed? [y/N] ").lower() != "y":
        logging.info("Execution aborted by the user!")
        return

    for i, impath in enumerate(tqdm(images)):
        np.random.seed(1)
        cv2.setRNGSeed(0)
        im = imread(impath)
        im = utils.rescale(im, args.rescale, channel_axis=2)

        chan, stats, contour, bin_im = mothseg.segment(im, method=args.method)

        pois = None
        if args.pois:
            cont_arr = np.zeros_like(bin_im)
            cont_arr = cv2.drawContours(cont_arr, [contour], 0, 1, -1)
            pois = mothseg.PointsOfInterest.detect(cont_arr)

        if args.calibration:
            positions = {pos.name.lower(): pos for pos in scalebar.Position}
            pos = positions[args.calib_pos]

            crop_x, crop_y = size = (args.calib_rel_width, args.calib_rel_height)
            cal_length, interm = scalebar.get_scale(im, 
                                                    cv2_corners=True,
                                                    pos=pos,
                                                    crop_size=size, 
                                                    return_intermediate=True,
                                                    binarize=True,
                                          )
            if args.show_interm:
                vis.plot_interm(im, interm, cal_length)
            
            if cal_length is not None and cal_length > 0:
                stats['c-area-calibrated'] = stats['c-area'] / cal_length ** 2
                stats['width-calibrated'] = (stats['c-xmax'] - stats['c-xmin']) / cal_length
                stats['height-calibrated'] = (stats['c-ymax'] - stats['c-ymin']) / cal_length
                stats['calibration-length'] = cal_length

        print(stats)
        #vis.plot([im, bin_im, chan], contour, stats, pois=pois)
        


main(parser.parse_args())