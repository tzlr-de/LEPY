from dataclasses import dataclass
from dataclasses import field

from lepy.outputs.definitions import Size
from lepy.outputs.definitions import Statistic
from lepy.outputs.definitions import Contour
from lepy.outputs.definitions import Calibration
from lepy.outputs.definitions import PointsOfInterest
from lepy.outputs.definitions import Box
from lepy.outputs.definitions import Coord
from lepy.outputs.definitions import POIDist
from lepy.outputs.definitions import POIArea


def _default(factory):
    return field(default_factory=factory)

@dataclass
class Outputs:
    image: Size = _default(lambda: Size(width="image_width", height="image_height"))
    intensity: Statistic = _default(lambda: Statistic.new("intensity"))
    saturation: Statistic = _default(lambda: Statistic.new("saturation"))
    hue: Statistic = _default(lambda: Statistic.new("hue"))

    luminance: Statistic = _default(lambda: Statistic.new("luminance"))
    chromaticity_red: Statistic = _default(lambda: Statistic.new("chrom_red", display_name="Chromaticity red"))
    chromaticity_green: Statistic = _default(lambda: Statistic.new("chrom_green", display_name="Chromaticity green"))
    chromaticity_blue: Statistic = _default(lambda: Statistic.new("chrom_blue", display_name="Chromaticity blue"))
    chromaticity_uv: Statistic = _default(lambda: Statistic.new("chrom_uv", display_name="Chromaticity UV"))

    red: Statistic = _default(lambda: Statistic.new("red"))
    green: Statistic = _default(lambda: Statistic.new("green"))
    blue: Statistic = _default(lambda: Statistic.new("blue"))
    uv: Statistic = _default(lambda: Statistic.new("uv", display_name="UV"))
    rgbuv: Statistic = _default(lambda: Statistic.new("rgbuv", display_name="RGB-UV mixed"))

    contour: Contour = _default(lambda: Contour())
    calibration: Calibration = _default(lambda: Calibration())
    poi: PointsOfInterest = _default(lambda: PointsOfInterest())

    def key2name(self, key: str) -> str:
        return {
            self.image.width: "Image width",
            self.image.height: "Image height",
            **self.intensity.key2name,
            **self.saturation.key2name,
            **self.hue.key2name,
            **self.red.key2name,
            **self.green.key2name,
            **self.blue.key2name,
            **self.uv.key2name,
            **self.rgbuv.key2name,

            self.contour.area: "Contour area",
            self.contour.width_calibrated: "Wing span",
            self.contour.height_calibrated: "Body length",

            self.calibration.length: "Calibration length",

            self.poi.area.body: "Body area",
            self.poi.area.wing_l: "Left wing area",
            self.poi.area.wing_r: "Right wing area",
            self.poi.dist.body: "Body length",
            self.poi.dist.inner: "Body width",
            self.poi.dist.inner_outer_l: "Left FWL",
            self.poi.dist.inner_outer_r: "Right FWL",
        }.get(key, key)

OUTPUTS = Outputs()

__all__ = [
    "OUTPUTS",
    "Size",
    "Statistic",
    "Contour",
    "Calibration",
    "PointsOfInterest",
    "Box",
    "Coord",
    "POIDist",
    "POIArea",
]
