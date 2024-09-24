from dataclasses import dataclass

from mothseg.output_definitions import Size
from mothseg.output_definitions import Statistic
from mothseg.output_definitions import Contour
from mothseg.output_definitions import Calibration
from mothseg.output_definitions import PointsOfInterest


@dataclass
class Outputs:
    image: Size = Size(width="image_width", height="image_height")
    intensity: Statistic = Statistic(
        median="median_intensity",
        mean="mean_intensity",
        stddev="stddev_intensity")
    saturation: Statistic = Statistic(
        median="median_saturation",
        mean="mean_saturation",
        stddev="stddev_saturation")
    hue: Statistic = Statistic(
        median="median_hue",
        mean="mean_hue",
        stddev="stddev_hue")

    contour: Contour = Contour()
    calibration: Calibration = Calibration()
    poi: PointsOfInterest = PointsOfInterest()

    def key2name(self, key: str) -> str:
        return {
            self.image.width: "Image width",
            self.image.height: "Image height",
            self.intensity.median: "Median intensity",
            self.intensity.mean: "Mean intensity",
            self.intensity.stddev: "Std. dev. intensity",
            self.saturation.median: "Median saturation",
            self.saturation.mean: "Mean saturation",
            self.saturation.stddev: "Std. dev. saturation",
            self.hue.median: "Median hue",
            self.hue.mean: "Mean hue",
            self.hue.stddev: "Std. dev. hue",

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
