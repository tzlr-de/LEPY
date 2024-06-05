#!/usr/bin/env python

if __name__ != '__main__':
    raise Exception("Do not import me!")

from pathlib import Path
import cv2
import numpy as np
import scalebar
import logging


from multiprocessing import Pool
from matplotlib.pyplot import imread
from tqdm.auto import tqdm
from scalebar.core.size import Size

import mothseg
from mothseg import parser
from mothseg import utils


def proceed_check(yes: bool) -> bool:
    return yes or input("Do you want to proceed? [y/N] ").lower() == "y"


class Worker:
    def __init__(self, config, plotter: mothseg.OutputWriter, *,
                progress_callback=None,
                raise_on_error=False):

        self.config = config
        self.plotter = plotter
        self.progress_callback = progress_callback
        self.raise_on_error = raise_on_error

    def callback(self, *args, **kwargs):
        if self.progress_callback is not None and callable(self.progress_callback):
            return self.progress_callback(*args, **kwargs)

    def __call__(self, impath):
            try:
                return impath, self.process(impath), None
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if self.raise_on_error:
                    raise

                return impath, None, e

    def process(self, impath: str):

        np.random.seed(1)
        cv2.setRNGSeed(0)
        imname = Path(impath).name

        self.callback(f"[{imname}] Loading image")
        im = imread(impath)

        # orig_im_H, orig_im_W, *_ = im.shape
        self.callback(f"[{imname}] Segmenting image ({im.shape[1]}x{im.shape[0]} px)")
        chan, stats, contour, bin_im = mothseg.segment(im, rescale=0.5, method=self.config.segmentation.method)
        # stats.update({"orig-image-width": orig_im_W, "orig-image-height": orig_im_H, "rescale-factor": rescale})

        calib_config = self.config.calibration
        if calib_config.enabled:
            self.callback(f"[{imname}] Calibrating image")
            res = scalebar.Result.new(impath,
                                    scalebar_size=Size.get(calib_config.scalebar_size),
                                    max_corners=50,
                                    size_per_square=calib_config.square_size,
                                    scalebar_location=scalebar.Position.get(calib_config.position),
                                    )

            self.plotter.plot_interm(impath, result=res)

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
        if self.config.points_of_interest.enabled:
            self.callback(f"[{imname}] Detecting points of interest")
            cont_arr = np.zeros_like(bin_im)
            cont_arr = cv2.drawContours(cont_arr, [contour], 0, 1, -1)
            pois = mothseg.PointsOfInterest.detect(cont_arr)

            stats.update(pois.stats)
            stats.update(pois.distances(scale=scale))


        self.callback(f"[{imname}] Plotting results")
        self.plotter.plot(impath, [im, bin_im, chan], contour, stats, pois=pois)

        return stats


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

    if args.n_jobs > 1 or args.n_jobs == -1:
        N = args.n_jobs if args.n_jobs > 0 else None
        pool = Pool(N)
        mapper = pool.imap
        logging.info(f"Initialized worker with {pool._processes} processes...")
    else:
        pool = None
        mapper = map

    logging.info("Starting processing...")
    with tqdm(total=len(images)) as bar:

        writer = mothseg.OutputWriter(output, config=args.config)
        plotter = mothseg.Plotter(output, plot_interm=args.plot_interm)
        worker = Worker(config, plotter,
                        progress_callback=None, #bar.set_description,
                        raise_on_error=args.raise_on_error)

        for impath, stats, e in mapper(worker, images):
            bar.set_description(f"Processing {Path(impath).name}")
            if e is not None:
                writer.log_fail(impath, e)
                continue
            assert stats is not None
            writer(impath, stats)
            bar.update(1)

    if pool is not None:
        logging.info("Waiting for all processes to finish...")
        pool.close()
        pool.join()

    logging.info("All done!")


exit(main(parser.parse_args()))
