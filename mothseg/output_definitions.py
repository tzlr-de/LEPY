import numpy as np

from dataclasses import dataclass

@dataclass
class Coord:
    x: str
    y: str

@dataclass
class Size:
    width: str
    height: str

@dataclass
class Box:
    x: str
    y: str
    w: str
    h: str

@dataclass
class Statistic:
    median: str
    mean: str
    stddev: str

    def calc_stats(self, arr) -> dict:
        return {
            self.median: float(np.median(arr)),
            self.mean: float(np.mean(arr)),
            self.stddev: float(np.std(arr)),
        }

@dataclass
class POIDist:
    inner_outer_l: str = "poi_dist_inner_outer_l"
    inner_outer_r: str = "poi_dist_inner_outer_r"
    inner: str = "poi_dist_inner"
    body: str = "poi_dist_body"

@dataclass
class POIArea:
    body: str = "poi_area_body"
    wing_l: str = "poi_area_wing_l"
    wing_r: str = "poi_area_wing_r"

@dataclass
class PointsOfInterest:
    orig_width: str = "poi_orig_width"
    orig_height: str = "poi_orig_height"
    dist: POIDist = POIDist()
    area: POIArea = POIArea()
    center: Coord = Coord(x="poi_center_x", y="poi_center_y")
    body_top: Coord = Coord(x="poi_body_top_x", y="poi_body_top_y")
    body_bot: Coord = Coord(x="poi_body_bot_x", y="poi_body_bot_y")
    outer_l: Coord = Coord(x="poi_outer_l_x", y="poi_outer_l_y")
    outer_r: Coord = Coord(x="poi_outer_r_x", y="poi_outer_r_y")
    inner_top_l: Coord = Coord(x="poi_inner_top_l_x", y="poi_inner_top_l_y")
    inner_top_r: Coord = Coord(x="poi_inner_top_r_x", y="poi_inner_top_r_y")
    inner_bot_l: Coord = Coord(x="poi_inner_bot_l_x", y="poi_inner_bot_l_y")
    inner_bot_r: Coord = Coord(x="poi_inner_bot_r_x", y="poi_inner_bot_r_y")

@dataclass
class Contour:
    length: str = "contour_length"
    area: str = "contour_area"
    xmin: str = "contour_xmin"
    xmax: str = "contour_xmax"
    ymin: str = "contour_ymin"
    ymax: str = "contour_ymax"
    area_calibrated: str = "contour_area_calibrated"
    width_calibrated: str = "contour_width_calibrated"
    height_calibrated: str = "contour_height_calibrated"


@dataclass
class Calibration:
    length: str = "calibration_length"
    pos: Box = Box(x="calibration_pos_x", y="calibration_pos_y", w="calibration_pos_w", h="calibration_pos_h")
