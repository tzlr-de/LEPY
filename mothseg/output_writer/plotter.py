import scalebar
import numpy as np
import cv2

from matplotlib import pyplot as plt

from mothseg import PointsOfInterest
from mothseg import visualization as vis
from mothseg.image import ColorStats
from mothseg.image import Image
from mothseg.output_writer.base import BaseWriter
from mothseg.outputs import OUTPUTS as OUTS


def _plot_images(ims, titles, grid, *, row: int, mask=None, cmaps=None):
    if cmaps is None:
        cmaps = [None] * len(ims)

    suffix = "\n(masked)"
    if mask is None:
        mask = 1
        suffix = ""

    for i, (chan, title, cmap) in enumerate(zip(ims, titles, cmaps)):
        ax = plt.subplot(grid[row, i])
        if chan.ndim == 2:
            ax.imshow(chan * mask, cmap="gray" if cmap is None else cmap)
        else:
            ax.imshow(chan * mask)

        title_color = "black"
        if isinstance(title, tuple):
            title, title_color = title
        _title_obj = ax.set_title(f"{title}{suffix}")
        _title_obj.set_color(title_color)
        ax.axis("off")

def _plot_histograms(ax, col_stats, *, colors, titles, alpha=0.3):
    for hist, col, title in zip(col_stats.histograms, colors, titles):
        if hist is None:
            continue
        ax.plot(col_stats.bins[:-1], hist, color=col, label=title)
        ax.fill_between(col_stats.bins[:-1], hist, color=col, alpha=alpha)

    ax.set_xlim(-1, 256)
    ax.set_xticks(np.linspace(0, 256, 5))
    ax.set_xticks(np.linspace(0, 256, 5*5), minor=True)
    ax.legend()

def _plot_boxplots(ax, channels, *, colors, mask,
                   median_color="yellow",
                   width=0.7,
                   showfliers=False,
                   ):
    for i, (chan, col) in enumerate(zip(channels, colors)):
        ax.boxplot(chan[mask != 0], patch_artist=True,
                   vert=False,
                   widths=width,
                   boxprops=dict(facecolor=col),
                   medianprops=dict(color=median_color),
                   positions=[len(channels) - i],
                   showfliers=showfliers)

    ax.set_xlim(-1, 256)
    ax.set_xticks(np.linspace(0, 256, 5))
    ax.set_xticks(np.linspace(0, 256, 5*5), minor=True)
    ax.set_yticklabels(colors)

def _plot_col_stats(ax, stats):
    ax.axis("off")

    rows = [[int(q25), int(median), int(q75), int(iqr)] for _, q25, q75, median, iqr in stats]
    tab  = ax.table(cellText=rows,
                colLabels=[
                    "Q25", "Median", "Q75", "IQR"],
                colLoc="center",
                cellLoc="center",
                loc="center",
                edges="horizontal",
                fontsize=72,
                )
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1, 1.5)

def _plot_stats(ax, stats):
    ax.axis("off")
    rows = []
    for key in [OUTS.poi.dist.inner_outer_l, OUTS.poi.dist.inner_outer_r,
                OUTS.poi.dist.inner, OUTS.poi.dist.body,
                OUTS.contour.width_calibrated,
                OUTS.poi.area.wing_l, OUTS.poi.area.wing_r, OUTS.poi.area.body]:
        value = stats.get(key, None)
        unit = "mm²" if "area" in key else "mm"
        rows.append([OUTS.key2name(key), f"{value:.2f} {unit}"])

    tab  = ax.table(cellText=rows,
                colLoc="center",
                cellLoc="left",
                loc="center",
                edges="horizontal",
                fontsize=72,
                )
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1, 1.5)

def _plot_traits(ax, mask, contour, pois, stats):
    ax.axis("off")

    ax.imshow(mask)
    ax.plot(contour[:, 0], contour[:, 1], linewidth=2, alpha=0.6)

    x0, y0, x1, y1 = [stats[coord] for coord in [OUTS.contour.xmin, OUTS.contour.ymin, OUTS.contour.xmax, OUTS.contour.ymax]]
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=None,
                               edgecolor='m', linewidth=2))

    ax.annotate("", xy=(x0, y1), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='<->'))

    if OUTS.calibration.length not in stats:
        unit = "px"
        lengthx = '{:.0f} pixels'.format(x1 - x0)
    else:
        unit = "mm"
        lengthx = '{:.2f} mm'.format(stats[OUTS.contour.width_calibrated])

    ax.text(0.5 * (x1 + x0), y1 + 20,
                lengthx,
                horizontalalignment='center', verticalalignment='top', fontsize=18)

    if pois is None:
        return

    for _, poi in pois:
        ax.scatter(poi.col, poi.row, color="red")

    for key, p0, p1 in pois.named_distances:
        dist = stats[key]
        ax.annotate("", xy=(p0.col, p0.row), xytext=(p1.col, p1.row),
                    arrowprops=dict(arrowstyle='<->'))
        ax.text(x=(p0.col + p1.col)/2, y=(p0.row + p1.row)/2 + 15,
                s=f"{dist:.2f} {unit}",
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=10,
        )

def _plot_scalebar(ax, calib_result):
    ax.axis("off")
    ax.imshow(calib_result.scalebar, cmap=plt.cm.gray)
    ax.set_title(f"Detected scalebar: \n{calib_result.scale:.2f} px/mm")

    if calib_result.distances is None:
        return

    corners = calib_result.distances.corners
    mask = np.zeros(len(corners), dtype=bool)
    selected_corners, pairs = calib_result.best_corners()
    mask[selected_corners] = True

    ys, xs = corners[mask].transpose(1, 0)
    ax.scatter(xs, ys, marker="o", c="blue")

    ys, xs = corners[~mask].transpose(1, 0)
    ax.scatter(xs, ys, marker="o", c="red", alpha=0.5)

class Plotter(BaseWriter):

    def __init__(self, folder: str, *, plot_interm: bool) -> None:
        super().__init__(folder)
        self._plot_interm = plot_interm

    def plot(self, image: Image, *,
             pois: PointsOfInterest,
             calib_result: scalebar.Result,
             col_stats: ColorStats
             ) -> None:

        dest = self.new_path(image.rgb_path, ".png", subfolder="visualizations")


        im, uv, mask = image.rgb_im, image.uv_im, image.mask

        gray = image.gray_im
        intensity_img = image.intensity_im

        fig = plt.figure(figsize=(16, 9))
        grid = plt.GridSpec(5, 7, figure=fig)

        saturation = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)[:, :, 1]
        saturation -= saturation.min()
        saturation = (saturation / saturation.max() * 255).astype(np.uint8)

        if image.has_uv:
            images = [im, uv, gray, saturation, mask]
            titles = ["RGB image", "UV image", "B/W image", "Saturation",  "Mask"]
        else:
            images = [im, gray, saturation, mask]
            titles = ["RGB image", "B/W image", "Saturation", "Mask"]

        _plot_images(ims=images, titles=titles, grid=grid, row=0)

        R, G, B = im.transpose(2, 0, 1)
        if image.has_uv:

            channels = [R, G, B, uv, intensity_img]
            colors = ["red", "green", "blue", "purple", "black"]
            titles = [ "Red channel", "Green channel", "Blue channel", "UV channel", "B/W + UV image"]
        else:
            channels = [R, G, B, intensity_img]
            colors = ["red", "green", "blue", "black"]
            titles = ["Red channel", "Green channel", "Blue channel", "B/W image"]

        _plot_images(ims=channels,
                     titles=list(zip(titles, colors)),
                     grid=grid,
                     row=1,
                     mask=mask)

        _plot_traits        (plt.subplot(grid[ :2, 5: ]), mask, image.contour, pois, image.stats)
        _plot_histograms    (plt.subplot(grid[2:4,  :5]), col_stats, colors=colors, titles=titles)
        _plot_boxplots      (plt.subplot(grid[4  ,  :5]), channels, colors=colors, mask=mask)
        _plot_stats         (plt.subplot(grid[2:4, 5: ]), image.stats)
        _plot_col_stats     (plt.subplot(grid[4  , 5  ]), col_stats)
        _plot_scalebar      (plt.subplot(grid[4  , 6  ]), calib_result)

        plt.tight_layout()

        if dest is not None:
            fig.savefig(dest)
        else:
            plt.show()
        plt.close()


    def plot_interm(self, impath: str, result: scalebar.Result):
        if not self._plot_interm:
            return
        dest = self.new_path(impath, ".png", subfolder="interm")
        fig = vis.plot_interm(result)
        if dest is not None:
            fig.savefig(dest)
        else:
            plt.show()
        plt.close()
