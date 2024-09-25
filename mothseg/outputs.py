from dataclasses import dataclass

from mothseg.output_definitions import Size
from mothseg.output_definitions import Statistic
from mothseg.output_definitions import Contour
from mothseg.output_definitions import Calibration
from mothseg.output_definitions import PointsOfInterest


@dataclass
class Outputs:
    image: Size = Size(width="image_width", height="image_height")
    intensity: Statistic = Statistic.new("intensity")
    saturation: Statistic = Statistic.new("saturation")
    hue: Statistic = Statistic.new("hue")

    red: Statistic = Statistic.new("red")
    green: Statistic = Statistic.new("green")
    blue: Statistic = Statistic.new("blue")
    uv: Statistic = Statistic.new("uv")
    black: Statistic = Statistic.new("black")

    contour: Contour = Contour()
    calibration: Calibration = Calibration()
    poi: PointsOfInterest = PointsOfInterest()

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

__all__ = ["OUTPUTS"]
