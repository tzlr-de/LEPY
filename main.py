#!/usr/bin/env python

if __name__ != '__main__':
    raise Exception("Do not import me!")

import cv2
import numpy as np
import scalebar
import logging


from matplotlib.pyplot import imread
from tqdm.auto import tqdm
from scalebar.core.size import Size

import mothseg
from mothseg import parser
from mothseg import utils


def proceed_check(yes: bool) -> bool:
    return yes or input("Do you want to proceed? [y/N] ").lower() == "y"

def callback(func, *args, **kwargs):
    if func is not None and callable(func):
        return func(*args, **kwargs)

def process(impath, config, writer: mothseg.OutputWriter, *,
            plot_interm: bool = False,
            progress_callback=None,
            ):

    np.random.seed(1)
    cv2.setRNGSeed(0)
    callback(progress_callback, "Loading image")
    im = imread(impath)
    # orig_im_H, orig_im_W, *_ = im.shape
    callback(progress_callback, f"Segmenting image ({im.shape[1]}x{im.shape[0]} px)")
    chan, stats, contour, bin_im = mothseg.segment(im, rescale=0.5, method=config.segmentation.method)
    # stats.update({"orig-image-width": orig_im_W, "orig-image-height": orig_im_H, "rescale-factor": rescale})

    calib_config = config.calibration
    if calib_config.enabled:
        callback(progress_callback, "Calibrating image")
        res = scalebar.Result.new(impath,
                                  scalebar_size=Size.get(calib_config.scalebar_size),
                                  max_corners=50,
                                  size_per_square=calib_config.square_size,
                                  scalebar_location=scalebar.Position.get(calib_config.position),
                                )
        if plot_interm:
            writer.plot_interm(impath, result=res)

        scale = res.scale
        if scale is not None and scale > 0:
            stats['calibration-length'] = scale
            stats['calibration-pos-x'] = int(res.position.x)
            stats['calibration-pos-y'] = int(res.position.y)
            stats['calibration-pos-w'] = int(res.position.width)
            stats['calibration-pos-h'] = int(res.position.height)
            stats['contour-area-calibrated'] = stats['contour-area'] / scale ** 2
            stats['width-calibrated'] = (stats['contour-xmax'] - stats['contour-xmin']) / scale
            stats['height-calibrated'] = (stats['contour-ymax'] - stats['contour-ymin']) / scale

    pois = None
    if config.points_of_interest.enabled:
        callback(progress_callback, "Detecting points of interest")
        cont_arr = np.zeros_like(bin_im)
        cont_arr = cv2.drawContours(cont_arr, [contour], 0, 1, -1)
        pois = mothseg.PointsOfInterest.detect(cont_arr)

        stats.update(pois.stats)
        stats.update(pois.distances(scale=scale))

    writer(impath, stats)
    callback(progress_callback, "Plotting results")
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

    with tqdm(images) as bar:
        for i, impath in enumerate(bar):
            try:
                process(impath, config, writer=writer,
                        plot_interm=args.plot_interm,
                        progress_callback=bar.set_description,
                    )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if args.raise_on_error:
                    raise
                writer.log_fail(impath, err=e)



exit(main(parser.parse_args()))
