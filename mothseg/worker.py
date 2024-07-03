import cv2
import numpy as np
import scalebar

from pathlib import Path
from scalebar.core.size import Size
from matplotlib.pyplot import imread

from mothseg.poi import PointsOfInterest
from mothseg.segmentation import segment
from mothseg.output_writer import Plotter

class Worker:
    def __init__(self, config, plotter: Plotter = None, *,
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
        chan, stats, contour, bin_im = segment(im, rescale=0.5, method=self.config.segmentation.method)
        # stats.update({"orig-image-width": orig_im_W, "orig-image-height": orig_im_H, "rescale-factor": rescale})

        calib_config = self.config.calibration
        res = None
        if calib_config.enabled:
            self.callback(f"[{imname}] Calibrating image")

            scalebar_location = scalebar.Position.get(calib_config.position) if "position" in calib_config else None
            res = scalebar.Result.new(impath,
                                    scalebar_size=Size.get(calib_config.scalebar_size),
                                    max_corners=50,
                                    size_per_square=calib_config.square_size,
                                    scalebar_location=scalebar_location,
                                    )

            if self.plotter is not None:
                self.plotter.plot_interm(impath, result=res)

            scale = float(res.scale)
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
            pois = PointsOfInterest.detect(cont_arr)

            stats.update(pois.stats)
            stats.update(pois.distances(scale=scale))
            stats.update(pois.areas(bin_im=cont_arr, scale=scale))


        self.callback(f"[{imname}] Plotting results")
        if self.plotter is not None:
            self.plotter.plot(impath, [im, bin_im, chan], contour, stats,
                              pois=pois,
                              calib_result=res,
                              )

        return stats
