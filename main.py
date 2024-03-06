#!/usr/bin/env python

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


def process(impath, config, writer: mothseg.OutputWriter, *, 
            rescale: float,
            show_interm: bool = False
            ):

    np.random.seed(1)
    cv2.setRNGSeed(0)
    im = imread(impath)
    orig_im_H, orig_im_W, *_ = im.shape
    im = utils.rescale(im, rescale, channel_axis=2)

    chan, stats, contour, bin_im = mothseg.segment(im, method=config.segmentation.method)
    stats.update({"orig-image-width": orig_im_W, "orig-image-height": orig_im_H, "rescale-factor": rescale})

    cal_length = None
    if config.calibration.enabled:
        positions = {pos.name.lower(): pos for pos in scalebar.Position}
        if config.calibration.position == "auto":
            pos = scalebar.Position.estimate(im)
            logging.debug(f"Estimated position: {pos.name}")
        elif config.calibration.position not in positions:
            raise ValueError(f"Unsupported position: {config.calibration.position}")
        else:
            pos = positions.get(config.calibration.position)
        size = (config.calibration.rel_width, config.calibration.rel_height)
        cal_length, interm = scalebar.get_scale(im, 
                                                cv2_corners=True,
                                                pos=pos,
                                                crop_size=size, 
                                                return_intermediate=True,
                                                binarize=True,
                                        )
        if show_interm:
            writer.plot_interm(impath, im, interm, cal_length)
        
        if cal_length is not None and cal_length > 0:
            stats['c-area-calibrated'] = stats['c-area'] / cal_length ** 2
            stats['width-calibrated'] = (stats['c-xmax'] - stats['c-xmin']) / cal_length
            stats['height-calibrated'] = (stats['c-ymax'] - stats['c-ymin']) / cal_length
            stats['calibration-length'] = cal_length
            stats['calibration-position'] = pos.name.lower()

    pois = None
    if config.points_of_interest.enabled:
        cont_arr = np.zeros_like(bin_im)
        cont_arr = cv2.drawContours(cont_arr, [contour], 0, 1, -1)
        pois = mothseg.PointsOfInterest.detect(cont_arr)

        stats.update(pois.stats)
        stats.update(pois.distances(cal_length=cal_length))

    writer(impath, stats)
    writer.plot(impath, [im, bin_im, chan], contour, stats, pois=pois)


def main(args):
    config = utils.read_config(args.config)
    images = utils.find_images(args.folder)

    logging.info(f"Found {len(images):,d} images in {args.folder}")
    first10 = '\n'.join(map(str, images[:10]))
    logging.debug(f"Here are the first 10: \n{first10}")

    if not proceed_check(args.yes):
        logging.info("Execution aborted by the user!")
        return -1
    
    output = utils.check_output(args, use_timestamp=args.use_timestamp)

    if output is None:
        logging.info("No output folder selected, exiting script.")
        return -2
    else:
        if not proceed_check(args.yes):
            logging.info("Execution aborted by the user!")
            return -1

    
    writer = mothseg.OutputWriter(output, config=args.config)

    for i, impath in enumerate(tqdm(images)):
        try:
            process(impath, config, writer=writer,
                    rescale=args.rescale,
                    show_interm=args.show_interm,
                )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            writer.log_fail(impath, err=e)



exit(main(parser.parse_args()))