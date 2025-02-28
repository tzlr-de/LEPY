import cv2
import numpy as np
import typing as T

from mothseg.outputs.writer import Plotter
from mothseg.image import Image

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

    def __call__(self, key_image: T.Tuple[str, Image]):
        key, image = key_image
        np.random.seed(1)
        cv2.setRNGSeed(0)
        try:
            return key, self.process(image), None
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if self.raise_on_error:
                raise

            return key, None, e

    def process(self, image: Image):
        key = image.key

        self.callback(f"[{key}] Loading image")
        image.read(uv_channel_index=self.config.reading.uv_channel_index,)
        im = image.rgb_im

        self.callback(f"[{key}] Segmenting image ({im.shape[1]}x{im.shape[0]} px)")
        image.segment(config=self.config.segmentation)

        self.callback(f"[{key}] Calibrating image")
        calib_result = image.calibrate(self.config.calibration)
        scale = None
        if calib_result is None:
            self.callback(f"[{key}] Calibration was not enabled")
        else:
            scale = calib_result.scale
            if self.plotter is not None:
                self.plotter.plot_interm(image.rgb_path, result=calib_result)

        self.callback(f"[{key}] Detecting points of interest")
        pois = image.pois(self.config.points_of_interest, scale=scale)

        self.callback(f"[{key}] Plotting results")
        if self.plotter is not None:
            self.plotter.plot(image,
                              pois=pois,
                              calib_result=calib_result,
                            )
            if image.has_uv:
                self.plotter.save_img(image.rgb_path, image.gbuv_im, subfolder="gbuv")

        return image.stats
