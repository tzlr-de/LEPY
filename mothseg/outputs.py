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

OUTPUTS = Outputs()

__all__ = ["OUTPUTS"]
