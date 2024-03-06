from __future__ import annotations

import numpy as np
import scipy as sp

from skimage import measure
from dataclasses import dataclass

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
    outer_l: Point
    outer_r: Point
    inner_l: Point
    inner_r: Point

    def __iter__(self):
        yield "center", self.center
        yield "outer_l", self.outer_l
        yield "outer_r", self.outer_r
        yield "inner_l", self.inner_l
        yield "inner_r", self.inner_r
    
    @property
    def stats(self):
        return {
            "poi-orig_width": int(self.width),
            "poi-orig_height": int(self.height),

            "poi-center-x": int(self.center.col),
            "poi-center-y": int(self.center.row),
            
            "poi-outer_l-x": int(self.outer_l.col),
            "poi-outer_l-y": int(self.outer_l.row),

            "poi-outer_r-x": int(self.outer_r.col),
            "poi-outer_r-y": int(self.outer_r.row),
            
            "poi-inner_l-x": int(self.inner_l.col),
            "poi-inner_l-y": int(self.inner_l.row),
            
            "poi-inner_r-x": int(self.inner_r.col),
            "poi-inner_r-y": int(self.inner_r.row),
        }

    def distances(self, im_width: int = None, im_height: int = None, cal_length: float = None):
        if cal_length is None:
            cal_length = 1.0
        if im_width is None:
            im_width = self.width
        if im_height is None:
            im_height = self.height

        w_scale = im_width / self.width / cal_length
        h_scale = im_height / self.height / cal_length


        res = {}
        for key, *pts in self.named_distances:
            p0, p1 = [p.rescale(width_scale=w_scale, height_scale=h_scale) for p in pts]
            res[key] = p0.dist(p1)
        return res

        # center = self.center.rescale(width_scale=w_scale, height_scale=h_scale)
        # outer_l = self.outer_l.rescale(width_scale=w_scale, height_scale=h_scale)
        # outer_r = self.outer_r.rescale(width_scale=w_scale, height_scale=h_scale)
        # inner_l = self.inner_l.rescale(width_scale=w_scale, height_scale=h_scale)
        # inner_r = self.inner_r.rescale(width_scale=w_scale, height_scale=h_scale)

    @property
    def named_distances(self):
        return [
            ("poi-dist-inner", self.inner_l, self.inner_r),
            ("poi-dist-inner-outer_l", self.inner_l, self.outer_l),
            ("poi-dist-inner-outer_r", self.inner_r, self.outer_r),
            ("poi-dist-center-outer_l", self.center, self.outer_l),
            ("poi-dist-center-outer_r", self.center, self.outer_r),
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
        inner_pix_l = detect_inner_pix(without_antenna_l, outer_pix_l, 'l')
        inner_pix_l = inner_pix_l + Point(0, outer_pix_l.col)
    
        # Right wing
        body_center_r = Point(middle_y, 0)  # to calculate outer_pix_r correctly
        without_antenna_r = remove_antenna(binary_right)
        outer_pix_r = detect_outer_pix(without_antenna_r, body_center_r)
        inner_pix_r = detect_inner_pix(without_antenna_r, outer_pix_r, 'r')
        inner_pix_r = inner_pix_r + Point(0, middle)
        outer_pix_r = outer_pix_r + Point(0, middle)

        return PointsOfInterest(
            width=W, height=H,
            center=body_center,
            outer_l=outer_pix_l,
            outer_r=outer_pix_r,
            inner_l=inner_pix_l,
            inner_r=inner_pix_r,
        )


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


