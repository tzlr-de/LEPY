from dataclasses import dataclass
from dataclasses import field

from mothseg.outputs.definitions import Size
from mothseg.outputs.definitions import Statistic
from mothseg.outputs.definitions import Contour
from mothseg.outputs.definitions import Calibration
from mothseg.outputs.definitions import PointsOfInterest
from mothseg.outputs.definitions import Box
from mothseg.outputs.definitions import Coord
from mothseg.outputs.definitions import POIDist
from mothseg.outputs.definitions import POIArea




@dataclass
class Outputs:
    image: Size = field(default_factory=lambda: Size(width="image_width", height="image_height"))
    intensity: Statistic = field(default_factory=lambda: Statistic.new("intensity"))
    saturation: Statistic = field(default_factory=lambda: Statistic.new("saturation"))
    hue: Statistic = field(default_factory=lambda: Statistic.new("hue"))

    red: Statistic = field(default_factory=lambda: Statistic.new("red"))
    green: Statistic = field(default_factory=lambda: Statistic.new("green"))
    blue: Statistic = field(default_factory=lambda: Statistic.new("blue"))
    uv: Statistic = field(default_factory=lambda: Statistic.new("uv"))
    black: Statistic = field(default_factory=lambda: Statistic.new("black"))

    contour: Contour = field(default_factory=lambda: Contour())
    calibration: Calibration = field(default_factory=lambda: Calibration())
    poi: PointsOfInterest = field(default_factory=lambda: PointsOfInterest())

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
            **self.black.key2name,

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
