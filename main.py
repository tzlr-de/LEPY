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

from mothseg import parser
from mothseg import utils


def proceed_check(yes: bool) -> bool:
    return yes or input("Do you want to proceed? [y/N] ").lower() == "y"

def main(args):
    config = utils.read_config(args.config)
    images = utils.find_images(args.folder)

    logging.info(f"Found {len(images):,d} images in {args.folder}")
    first10 = '\n'.join(map(str, images[:10]))
    logging.debug(f"Here are the first 10: \n{first10}")

    if not proceed_check(args.yes):
        logging.info("Execution aborted by the user!")
        return -1
    
    output = utils.check_output(args.output)

    if output is None:
        logging.info("No output folder selected, exiting script.")
        return -2
    else:
        logging.info(f"Outputs are stored to {output}.") 
        if not proceed_check(args.yes):
            logging.info("Execution aborted by the user!")
            return -1


    writer = mothseg.OutputWriter(output)

    for i, impath in enumerate(tqdm(images)):
        np.random.seed(1)
        cv2.setRNGSeed(0)
        im = imread(impath)
        im = utils.rescale(im, args.rescale, channel_axis=2)

        chan, stats, contour, bin_im = mothseg.segment(im, method=config.segmentation.method)

        pois = None
        if config.points_of_interest.enabled:
            cont_arr = np.zeros_like(bin_im)
            cont_arr = cv2.drawContours(cont_arr, [contour], 0, 1, -1)
            pois = mothseg.PointsOfInterest.detect(cont_arr)

            stats.update(pois.stats)

        if config.calibration.enabled:
            positions = {pos.name.lower(): pos for pos in scalebar.Position}
            pos = positions.get(config.calibration.position)
            if pos is None:
                logging.error(f"Unsupported position: {config.calibration.position}")
                return -3

            crop_x, crop_y = size = (config.calibration.rel_width, config.calibration.rel_height)
            cal_length, interm = scalebar.get_scale(im, 
                                                    cv2_corners=True,
                                                    pos=pos,
                                                    crop_size=size, 
                                                    return_intermediate=True,
                                                    binarize=True,
                                          )
            if args.show_interm:
                writer.plot_interm(impath, im, interm, cal_length)
            
            if cal_length is not None and cal_length > 0:
                stats['c-area-calibrated'] = stats['c-area'] / cal_length ** 2
                stats['width-calibrated'] = (stats['c-xmax'] - stats['c-xmin']) / cal_length
                stats['height-calibrated'] = (stats['c-ymax'] - stats['c-ymin']) / cal_length
                stats['calibration-length'] = cal_length

        writer(impath, stats)
        writer.plot(impath, [im, bin_im, chan], contour, stats, pois=pois)
        


exit(main(parser.parse_args()))