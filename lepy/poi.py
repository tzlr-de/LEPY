from __future__ import annotations

import numpy as np
import scipy as sp

from skimage import measure
from dataclasses import dataclass

from lepy.outputs import OUTPUTS as OUTS

@dataclass
class Point:
    row: int
    col: int

    def __radd__(self, other):
        return self+other

    def __add__(self, other):
        if isinstance(other, (tuple, list)):
            return Point(self.row + other[0], self.col + other[1])

        elif isinstance(other, Point):
            return Point(self.row + other.row, self.col + other.col)

        elif isinstance(other, np.ndarray):
            return other + self

    def __sub__(self, other):
        if isinstance(other, (tuple, list)):
            return Point(self.row - other[0], self.col - other[1])

        elif isinstance(other, Point):
            return Point(self.row - other.row, self.col - other.col)

        elif isinstance(other, np.ndarray):
            return self - other

    def rescale(self, width_scale: float, height_scale: float):
        return Point(self.row * width_scale, self.col * height_scale)

    def dist(self, other: Point):
        diff = self - other
        return np.sqrt(diff.row ** 2 + diff.col ** 2)

    def __array__(self, dtype=None):
        return np.array([self.row, self.col])


@dataclass
class PointsOfInterest:
    width: int
    height: int
    center: Point
    body_top: Point
    body_bot: Point
    outer_l: Point
    outer_r: Point
    inner_top_l: Point
    inner_top_r: Point
    inner_bot_l: Point
    inner_bot_r: Point

    def __iter__(self):
        """ used mainly for visualization """
        # yield "center", self.center
        yield "body_top", self.body_top
        yield "body_bot", self.body_bot
        yield "outer_l", self.outer_l
        yield "outer_r", self.outer_r
        yield "inner_top_l", self.inner_top_l
        yield "inner_top_r", self.inner_top_r
        yield "inner_bot_l", self.inner_bot_l
        yield "inner_bot_r", self.inner_bot_r

    @property
    def stats(self):
        return {
            OUTS.poi.orig_width: int(self.width),
            OUTS.poi.orig_height: int(self.height),

            OUTS.poi.center.x: int(self.center.col),
            OUTS.poi.center.y: int(self.center.row),

            OUTS.poi.body_top.x: int(self.body_top.col),
            OUTS.poi.body_top.y: int(self.body_top.row),

            OUTS.poi.body_bot.x: int(self.body_bot.col),
            OUTS.poi.body_bot.y: int(self.body_bot.row),

            OUTS.poi.outer_l.x: int(self.outer_l.col),
            OUTS.poi.outer_l.y: int(self.outer_l.row),

            OUTS.poi.outer_r.x: int(self.outer_r.col),
            OUTS.poi.outer_r.y: int(self.outer_r.row),

            OUTS.poi.inner_top_l.x: int(self.inner_top_l.col),
            OUTS.poi.inner_top_l.y: int(self.inner_top_l.row),

            OUTS.poi.inner_top_r.x: int(self.inner_top_r.col),
            OUTS.poi.inner_top_r.y: int(self.inner_top_r.row),

            OUTS.poi.inner_bot_l.x: int(self.inner_bot_l.col),
            OUTS.poi.inner_bot_l.y: int(self.inner_bot_l.row),

            OUTS.poi.inner_bot_r.x: int(self.inner_bot_r.col),
            OUTS.poi.inner_bot_r.y: int(self.inner_bot_r.row),

        }

    def distances(self, im_width: int = None, im_height: int = None, scale: float = None):
        if scale is None:
            scale = 1.0
        if im_width is None:
            im_width = self.width
        if im_height is None:
            im_height = self.height

        w_scale = im_width / self.width / scale
        h_scale = im_height / self.height / scale


        res = {}
        for key, *pts in self.named_distances:
            p0, p1 = [p.rescale(width_scale=w_scale, height_scale=h_scale) for p in pts]
            res[key] = float(p0.dist(p1))
        return res

    def areas(self, bin_im, scale: float = None):
        if scale is None:
            scale = 1.0
        res = {}
        for key, masks, val in self.named_areas(bin_im):
            res[key] = float(np.sum(masks == val) / scale ** 2)
        return res

    def named_areas(self, bin_im):

        res = bin_im.copy()

        body = res[:, self.inner_top_l.col:self.inner_top_r.col]
        body[body == 1] = 1

        left_wing = res[:, :self.inner_top_l.col]
        left_wing[left_wing == 1] = 2

        right_wing = res[:, self.inner_top_r.col:]
        right_wing[right_wing == 1] = 3

        return [
            (OUTS.poi.area.body, res, 1),
            (OUTS.poi.area.wing_l, res, 2),
            (OUTS.poi.area.wing_r, res, 3),
        ]

    @property
    def named_distances(self):
        return [
            (OUTS.poi.dist.inner, self.inner_top_l, self.inner_top_r),
            (OUTS.poi.dist.body, self.body_top, self.body_bot),
            (OUTS.poi.dist.inner_outer_l, self.inner_top_l, self.outer_l),
            (OUTS.poi.dist.inner_outer_r, self.inner_top_r, self.outer_r),
        ]

    @classmethod
    def detect(cls, bin_im: np.ndarray) -> PointsOfInterest:
        H, W, *C = bin_im.shape
        middle = split_picture(bin_im)

        binary_left = bin_im[:, :middle]
        binary_right = bin_im[:, middle:]

        # Centroid of central column
        middle_arr = bin_im[:, middle]
        middle_y = int(np.mean(np.argwhere(middle_arr)))
        body_center = Point(middle_y, middle)

        # Left wing
        without_antenna_l = remove_antenna(binary_left)
        outer_pix_l = detect_outer_pix(without_antenna_l, body_center)
        inner_top_l = detect_inner_pix(without_antenna_l, outer_pix_l, 'l')
        inner_top_l = inner_top_l + Point(0, outer_pix_l.col)
        inner_bot_l = detect_inner_bottom(without_antenna_l, inner_top_l, 'l')

        # Right wing
        body_center_r = Point(middle_y, 0)  # to calculate outer_pix_r correctly
        without_antenna_r = remove_antenna(binary_right)
        outer_r = detect_outer_pix(without_antenna_r, body_center_r)
        inner_top_r = detect_inner_pix(without_antenna_r, outer_r, 'r')
        inner_bot_r = detect_inner_bottom(without_antenna_r, inner_top_r, 'r')

        inner_top_r = inner_top_r + Point(0, middle)
        inner_bot_r = inner_bot_r + Point(0, middle)
        outer_r = outer_r + Point(0, middle)

        body_top, body_bot = detect_body(bin_im,
                                         inner_top_l, inner_top_r)


        return PointsOfInterest(
            width=W, height=H,
            center=body_center,
            body_top=body_top,
            body_bot=body_bot,
            outer_l=outer_pix_l,
            outer_r=outer_r,

            inner_top_l=inner_top_l,
            inner_top_r=inner_top_r,

            inner_bot_l=inner_bot_l,
            inner_bot_r=inner_bot_r,
        )

def detect_body(bin_im, inner_top_l, inner_top_r):
    body = bin_im[:, inner_top_l.col:inner_top_r.col]
    rows, cols = np.where(body == 1)
    col = int(np.mean(cols))
    middle = int(np.mean(rows))

    top = np.where(body[:middle, col] == 1)[0][0]
    bot = np.where(body[middle:, col] == 0)[0][0] + middle
    return Point(top, col + inner_top_l.col), Point(bot, col + inner_top_l.col)

def remove_antenna(half_binary):
    """Remove antenna if connected to the wing

    Arguments
    ---------
    half_binary : 2D array
        binary image of left/right wing

    Returns
    -------
    without_antenna : 2D array
        binary image, same shape as input without antenna (if it touches the
        wing)
    """
    markers, _ = sp.ndimage.label(
        1 - half_binary,
        sp.ndimage.generate_binary_structure(2, 1)
    )
    regions = measure.regionprops(markers)
    areas = np.array([r.area for r in regions])
    idx_sorted = 1 + np.argsort(-areas)[:2]

    try:
        dilated_bg = sp.ndimage.binary_dilation(
            markers == idx_sorted[0], iterations=35
        )
        dilated_hole = sp.ndimage.binary_dilation(
            markers == idx_sorted[1], iterations=35
        )
        intersection = np.minimum(dilated_bg, dilated_hole)
        without_antenna = np.copy(half_binary)
        without_antenna[intersection] = 0
    except IndexError:
        return half_binary

    return without_antenna


def detect_outer_pix(half_binary, center: Point) -> Point:
    """Relative (r, c) coordinates of outer pixel (wing's tip)

    Arguments
    ---------
    half_binary : 2D array
        Binary image of left/right wing.
    center : Point
        Centroid of the lepidopteran.

    Returns
    -------
    outer_pix : 1D array
        relative coordinates of the outer pixel (r, c)
    """
    markers, _ = sp.ndimage.label(
        half_binary,
        sp.ndimage.generate_binary_structure(2, 1)
    )
    regions = measure.regionprops(markers)
    areas = [r.area for r in regions]
    idx_max = np.argmax(areas)

    coords = np.array(regions[idx_max].coords)
    distances = np.linalg.norm(coords - center, axis=-1)
    idx_outer_pix = np.argmax(distances)
    outer_pix = coords[idx_outer_pix]

    return Point(*outer_pix)


def detect_inner_pix(half_binary, outer_pix: Point, side: str) -> Point:
    """Relative (r, c) coordinates of the inner pixel (between wing and body)

    Arguments
    ---------
    half_binary : 2D array
        binary image of left/right wing
    outer_pix : 1D array
        (r, c) coordinates (relative) of the outer pixel
    side : str
        left ('l') or right ('r') wing

    Returns
    -------
    inner_pix : 2D array
        relative coordinates of the inner pixel (r, c)

    Notes
    -----
    This function returns `inner_pix` from the following steps:
    1. Obtains `focus`, the region where to look for `inner_pix`;
    2. Gathers `focus_inv`, the inverse of `focus`, and returns the
       information on its regions;
    3. From the top region of `focus_inv`, the shoulder of the
       lepidopteran — the farthest point at the rows — is chosen
       as `inner_pix`.
    """
    lower_bound = int(half_binary.shape[0]*0.75)

    if side == 'l':
        focus = half_binary[:lower_bound, outer_pix.col:]
    else:
        focus = half_binary[:lower_bound, :outer_pix.col]

    focus_inv = 1 - focus

    markers, _ = sp.ndimage.label(
        focus_inv, sp.ndimage.generate_binary_structure(2, 1)
    )
    regions = measure.regionprops(markers)
    # if idx in regions is not 0, bottom region is considered for inner_pix,
    # instead of top region
    coords = regions[0].coords
    y_max = np.max(coords[:, 0])
    mask = (coords[:, 0] == y_max)
    selection = coords[mask]
    if side == 'l':
        idx = np.argmax(selection[:, 1])
    else:
        idx = np.argmin(selection[:, 1])

    inner_pix = selection[idx]

    return Point(*inner_pix)

def detect_inner_bottom(half_binary, inner_top: Point, side: str) -> Point:

    rows = np.where(1-half_binary[inner_top.row+1:, inner_top.col])[0]

    row, col = rows[0] + inner_top.row, inner_top.col
    return Point(row, col)

def split_picture(binary):
    """Calculate the middle of the butterfly.

    Parameters
    ----------
    binary : ndarray of bool
        Binary butterfly mask.

    Returns
    -------
    midpoint : int
        Horizontal coordinate for middle of butterfly.

    Notes
    -----
    Currently, this is calculated by finding the center
    of gravity of the butterfly.
    """
    column_weights = np.sum(binary, axis=0)
    column_weights_normalized = column_weights / np.sum(column_weights)
    column_idxs = np.arange(binary.shape[1])
    column_centroid = np.sum(column_weights_normalized * column_idxs)
    return int(column_centroid)
