import cv2
import numpy as np
import typing as T

from pathlib import Path
from dataclasses import dataclass
from dataclasses import field
from matplotlib.pyplot import imread

import scalebar
from mothseg.poi import PointsOfInterest
from mothseg.outputs import OUTPUTS as OUTS
from mothseg.output_definitions import Statistic


@dataclass
class ColorStats:
    histograms: T.List[np.ndarray]
    q25s: T.List[float]
    q75s: T.List[float]
    medians: T.List[float]
    iqrs: T.List[float]

    bins: np.ndarray

    def __iter__(self):
        return zip(self.histograms, self.q25s, self.q75s, self.medians, self.iqrs)

    def __getitem__(self, idx):
        return self.histograms[idx], self.q25s[idx], self.q75s[idx], self.medians[idx], self.iqrs[idx]

    def __len__(self):
        return len(self.histograms)


@dataclass
class Image:
    root: Path = None
    key: str = None
    rgb_fname: str = None
    uv_fname: str = None

    rgb_im: np.ndarray = None
    uv_im: np.ndarray = None

    gray_im: np.ndarray = None
    mask: np.ndarray = None
    chan: np.ndarray = None
    contour: np.ndarray = None

    stats: dict = field(default_factory=dict)

    @property
    def four_chan_im(self) -> np.ndarray:
        im, uv, mask = self.rgb_im, self.uv_im, self.mask
        if not self.has_uv:
            uv = np.zeros_like(im[..., 0])

        if mask is None:
            mask = np.ones(im.shape[:2], dtype=np.uint8)

        four_chan_im = np.zeros((*im.shape[:2], 4), dtype=im.dtype)
        four_chan_im[..., :3] = im * mask[..., None]
        four_chan_im[..., 3] = uv * mask if uv is not None else 0
        return four_chan_im

    @property
    def rgb_path(self) -> Path:
        return self.root / self.rgb_fname

    @property
    def uv_path(self) -> Path:
        return self.root / self.uv_fname

    @property
    def has_uv(self) -> bool:
        return self.uv_fname is not None

    @property
    def intensity_im(self) -> np.ndarray:

        gray_im, uv, mask = self.gray_im, self.uv_im, self.mask
        if mask is None:
            mask = np.ones(gray_im.shape, dtype=np.uint8)
        if uv is None:
            return gray_im * mask
        res = cv2.addWeighted(gray_im*mask, 0.5, uv*mask, 0.5, 0)
        return res

    def read(self, *, uv_channel_index: int = 0) -> None:

        if self.rgb_im is not None:
            return

        im = self.rgb_im = imread(self.rgb_path)
        self.gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        assert im.ndim == 3, f"Image {self.rgb_fname} is not a 3-channel image"
        if self.has_uv:
            self.uv_im = imread(self.uv_path)[..., uv_channel_index]
        else:
            self.uv_im = None

    def segment(self, config: T.Dict[str, T.Any] = None) -> dict:
        from mothseg.segmentation import segment # due to circular import
        im, uv = self.rgb_im, self.uv_im
        if config is None:
            self.mask = None
            return

        channel = config["channel"]
        if channel == "uv":
            assert uv is not None, "UV channel is not loaded"
            channel = uv.max() - uv

        self.chan, self.stats, self.contour, self.mask = \
            segment(im,
                    channel=channel,
                    rescale=config["rescale"],
                    method=config["method"],
                    ksize=config.get("ksize"),
                    fill_holes=config.get("fill_holes", True),
                    )


    def calibrate(self, config: T.Dict[str, T.Any]) -> T.Optional[scalebar.Result]:
        if not config["enabled"]:
            return

        scalebar_location = None
        if "position" in config and config.position is not None:
            scalebar_location = scalebar.Position.get(config.position)

        res = scalebar.Result.new(self.rgb_path,
                                  template_path=config.template_path,
                                  template_scale=config.template_scale,
                                  roi_fraction=config.roi_fraction,
                                  max_corners=config.max_corners,
                                  size_per_square=config.square_size,
                                  scalebar_location=scalebar_location,
                                )

        scale = float(res.scale)
        stats = self.stats
        if scale is not None and scale > 0:
            stats[OUTS.calibration.length] = scale
            stats[OUTS.calibration.pos.x] = int(res.position.x)
            stats[OUTS.calibration.pos.y] = int(res.position.y)
            stats[OUTS.calibration.pos.w] = int(res.position.width)
            stats[OUTS.calibration.pos.h] = int(res.position.height)
            stats[OUTS.contour.area_calibrated] = stats[OUTS.contour.area] / scale ** 2
            stats[OUTS.contour.width_calibrated] = (stats[OUTS.contour.xmax] - stats[OUTS.contour.xmin]) / scale
            stats[OUTS.contour.height_calibrated] = (stats[OUTS.contour.ymax] - stats[OUTS.contour.ymin]) / scale

        return res

    def pois(self, config: T.Dict[str, T.Any], scale: T.Optional[float] = None) -> T.Optional[PointsOfInterest]:
        if not config.enabled:
            return
        cont_arr = np.zeros_like(self.mask)
        cont_arr = cv2.drawContours(cont_arr, [self.contour], 0, 1, -1)
        pois = PointsOfInterest.detect(cont_arr)

        self.stats.update(pois.stats)
        self.stats.update(pois.distances(scale=scale))
        self.stats.update(pois.areas(bin_im=cont_arr, scale=scale))
        return pois


    def color_stats(self, *, binsize: int = 1) -> ColorStats:
        histograms, q25s, q75s, iqrs, medians = [], [], [], [], []

        bins = np.linspace(0, 255, 256//binsize, endpoint=True)
        def compute(channel, out_keys: Statistic):
            nonlocal histograms, q25s, q75s, iqrs, medians, binsize

            if self.mask is not None:
                channel = channel[self.mask != 0]

            # ... but we don't want to compute histogram for it
            # or store 0s in the output CSV table
            if (channel == 0).all():
                # histograms.append(None)
                self.stats.update({
                    out_keys.median: -1,
                    out_keys.Q25: -1,
                    out_keys.Q75: -1,
                    out_keys.IQR: -1,
                })

            else:
                hist, _ = np.histogram(channel, bins=bins)
                histograms.append(hist)

                medians.append(np.median(channel))
                q25s.append(np.percentile(channel, 25))
                q75s.append(np.percentile(channel, 75))
                iqrs.append(q75s[-1] - q25s[-1])

                self.stats.update({
                    out_keys.median: medians[-1],
                    out_keys.Q25: q25s[-1],
                    out_keys.Q75: q75s[-1],
                    out_keys.IQR: iqrs[-1],
                })

        for channel, out_keys in zip(self.four_chan_im.transpose(2, 0, 1), [OUTS.red, OUTS.green, OUTS.blue, OUTS.uv]):
            compute(channel, out_keys)

        compute(self.intensity_im, OUTS.black)

        return ColorStats(histograms=histograms,
                     q25s=q25s, q75s=q75s,
                     medians=medians,
                     iqrs=iqrs, bins=bins)
