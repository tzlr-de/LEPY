import scalebar
import numpy as np
import cv2
import git
import datetime as dt

from matplotlib import pyplot as plt

import lepy
from lepy import PointsOfInterest
from lepy import visualization as vis
from lepy.image import ColorStats
from lepy.image import Image
from lepy.outputs import OUTPUTS as OUTS
from lepy.outputs.writer.base import BaseWriter


def _plot_images(ims, titles, grid, *, row: int, mask=None, cmaps=None):
    if cmaps is None:
        cmaps = [None] * len(ims)

    suffix = "\n$masked$"
    if mask is None:
        mask = 1
        suffix = ""

    for i, (chan, title, cmap) in enumerate(zip(ims, titles, cmaps)):
        ax = plt.subplot(grid[row, i])
        if chan.ndim == 2:
            ax.imshow(chan * mask, cmap="gray" if cmap is None else cmap)
            if isinstance(mask, np.ndarray):
                _mask = mask.astype(np.float32)
                _mask[mask == 1] = np.nan
                ax.imshow(_mask, cmap="gray")
        else:
            ax.imshow(chan * mask)

        title_color = "black"
        if isinstance(title, tuple):
            title, title_color = title
        _title_obj = ax.set_title(f"{title}{suffix}")
        _title_obj.set_color(title_color)
        ax.axis("off")

def _plot_histograms(ax, histograms, bins, *, colors, titles,
                     alpha=0.3,
                     xlim=256, xticks=5, xticks_minor=5):
    assert len(colors) == len(titles) == len(histograms), "Invalid input sizes!"
    for hist, col, title in zip(histograms, colors, titles):
        if hist is None:
            continue
        ax.plot(bins, hist, color=col, label=title)
        ax.fill_between(bins, hist, color=col, alpha=alpha)

    ax.set_xlim(-1, xlim)
    ax.set_xticks(np.linspace(0, xlim, xticks))
    if xticks_minor is not None:
        ax.set_xticks(np.linspace(0, xlim, xticks*xticks_minor), minor=True)
    # ax.set_xlabel("Brightness")
    ax.legend()

def _plot_boxplots(ax, channels, *, colors, titles, mask,
                   median_color="yellow",
                   width=0.7,
                   showfliers=False,
                   xlim=256, xticks=5, xticks_minor=5
                   ):
    for i, (chan, col) in enumerate(zip(channels, colors)):
        ax.boxplot(chan[mask != 0], patch_artist=True,
                   vert=False,
                   widths=width,
                   boxprops=dict(facecolor="white" if col == "black" else col), # black is reserved for intensity
                   medianprops=dict(color=median_color),
                   positions=[len(channels) - i],
                   showfliers=showfliers)

    ax.set_xlim(-1, xlim)
    ax.set_xticks(np.linspace(0, xlim, xticks))
    if xticks_minor is not None:
        ax.set_xticks(np.linspace(0, xlim, xticks*xticks_minor), minor=True)
    ax.set_xlabel("Brightness")
    # ax.set_yticklabels([t.replace(" ", "\n") for t in titles])
    ax.set_yticklabels(titles)

def _plot_col_stats(ax, stats: ColorStats, *, colors, title):
    ax.axis("off")
    ax.set_title(title)

    rows = [[ int(q25), int(median), int(q75), int(iqr), f"{shannon:.2f}"]
                for _, q25, q75, median, iqr, shannon, _ in stats]
    tab  = ax.table(cellText=rows,
                    cellColours=
                    [
                        [(colors[i], 0.5) for _ in range(len(rows[0]))]
                            for i in range(0, len(rows))
                    ],
                    colLabels=["Q25", "Median\n$Brightness$", "Q75", "IQR\n$Contrast$", "Shannon Index"],
                    colLoc="center",
                    cellLoc="center",
                    loc="center",
                    edges="closed",
                    fontsize=72,
                )
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1, 1.5)


    cellDict = tab.get_celld()
    for i in range(len(rows[0])):
        cellDict[(0, i)].set_height(.35)

    for c in cellDict.values():
        c.set(linewidth=0)

    # for i in range(len(rows)):
    #     tab[i+1, 0].set_text_props(ha="left")

    # for j in range(0, len(rows[0])):
    #     for i in range(1, len(rows)):
    #         tab[i, j].visible_edges = "open"
    #     tab[len(rows), j].visible_edges = "B"

def _plot_stats(ax: plt.Axes, stats: dict, *, title: str):
    ax.axis("off")
    ax.set_title(title)
    rows = []
    for key in [OUTS.poi.dist.inner_outer_l, OUTS.poi.dist.inner_outer_r,
                OUTS.poi.dist.inner, OUTS.poi.dist.body,
                OUTS.contour.width_calibrated,
                OUTS.poi.area.wing_l, OUTS.poi.area.wing_r, OUTS.poi.area.body]:
        value = stats.get(key, None)
        unit = "mm²" if "area" in key else "mm"
        rows.append([OUTS.key2name(key), f"{value:.2f}", unit])

    tab  = ax.table(cellText=rows,
                colLoc="center",
                cellLoc="left",
                loc="center",
                edges="horizontal",
                fontsize=72,
                )
    for row in range(len(rows)):
        tab[row, 1].set_text_props(ha="right")
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1, 1.65)

    for col in range(0, len(rows[0])):
        tab[0, col].visible_edges = "T"
        for row in range(1, len(rows)):
            tab[row, col].visible_edges = "open"

        tab[4, col].visible_edges = "B"
        tab[len(rows)-1, col].visible_edges = "B"

def _plot_struc_traits(ax, mask, contour, pois, stats: dict):
    ax.axis("off")

    ax.imshow(mask)
    ax.plot(contour[:, 0], contour[:, 1], linewidth=2, alpha=0.6)

    x0, y0, x1, y1 = [stats[coord] for coord in [OUTS.contour.xmin, OUTS.contour.ymin, OUTS.contour.xmax, OUTS.contour.ymax]]
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=None,
                               edgecolor='m', linewidth=2))

    ax.annotate("", xy=(x0, y1), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='<->'))

    bbox_props = dict(boxstyle="round,pad=0.3", fc=("white", 0.5), ec="white", lw=2)
    if OUTS.calibration.length not in stats:
        unit = "px"
        lengthx = '{:.0f} pixels'.format(x1 - x0)
    else:
        unit = "mm"
        lengthx = '{:.2f} mm'.format(stats[OUTS.contour.width_calibrated])

    ax.text(0.5 * (x1 + x0), y1,
            lengthx,
            horizontalalignment='center',
            verticalalignment='top',
            fontsize=18,
            bbox=bbox_props,
    )

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
                bbox=bbox_props,
        )

def _plot_scalebar(ax, calib_result):
    # ax.axis("off")
    ax.grid(True)
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

def _add_meta_info(fig, ax, image_key: str, *, add_time: bool = False):
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha[:7]
    except git.InvalidGitRepositoryError:
        sha = "Not-a-git"
    now = dt.datetime.now()
    title = " | ".join(
        [   f"Lepy v{lepy.__version__}-{sha}",
            f"{image_key}",
            f"{now:%d.%m.%Y %H:%M:%S}" if add_time else f"{now:%d.%m.%Y}",
        ])
    # ax.set_title(title)
    fig.suptitle(title)


class Plotter(BaseWriter):

    def __init__(self, folder: str, *, plot_interm: bool, save_contours: bool = True) -> None:
        super().__init__(folder)
        self._plot_interm = plot_interm
        self._save_contours = save_contours

    def save_contour(self, image: Image, *, subfolder: str):
        """ stores relative contour coordinates to a file """
        if not self._save_contours:
            return

        path = self.new_path(image.rgb_path, ".txt", subfolder=subfolder)
        with open(path, "w") as f:
            f.write("0 ") # class index; we have only one class yet
            h, w = image.rgb_im.shape[:2]
            coords = []
            for x, y in image.contour:
                coords.append(f"{x / w:.7f} {y / h:.7f}")
            f.write(" ".join(coords))

    def plot(self, image: Image, *,
             pois: PointsOfInterest,
             calib_result: scalebar.Result,
             col_stat_binsize: int = 2,
             ) -> None:

        dest = self.new_path(image.rgb_path, ".png", subfolder="visualisations")
        im, uv, mask = image.rgb_im, image.uv_im, image.mask

        self.save_contour(image, subfolder="contours")

        gray = image.gray_im
        intensity_img = image.intensity_im

        fig = plt.figure(figsize=(16, 9))
        grid = plt.GridSpec(5, 8, figure=fig)
        col_stats = image.color_stats(binsize=col_stat_binsize)


        if image.has_uv:
            images = [im, uv, gray, image.gbuv_im, mask]
            titles = ["RGB image", "UV image", "B/W image", "GB-UV image",  "Mask"]
        else:
            saturation = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)[:, :, 1]
            saturation -= saturation.min()
            saturation = (saturation / saturation.max() * 255).astype(np.uint8)

            images = [im, gray, saturation, mask]
            titles = ["RGB image", "B/W image", "Saturation", "Mask"]

        _plot_images(ims=images, titles=titles, grid=grid, row=0)

        R, G, B = im.transpose(2, 0, 1)
        if image.has_uv:

            channels = [R, G, B, uv, intensity_img]
            colors = ["red", "green", "blue", "purple", "gray"]
            titles = [ "Red", "Green", "Blue", "UV", "RGB-UV mixed"]
        else:
            channels = [R, G, B, intensity_img]
            colors = ["red", "green", "blue", "gray"]
            titles = ["Red", "Green", "Blue", "B/W image"]

        _plot_images(ims=channels,
                     titles=list(zip(titles, colors)),
                     grid=grid,
                     row=1,
                     mask=mask)
        xtick_params = dict(xlim=255, xticks=6, xticks_minor=5)

        _plot_struc_traits  (plt.subplot(grid[ :2, 5:7 ]),
                             im, image.contour, pois, image.stats)
        _add_meta_info      (fig, plt.subplot(grid[ :2, 5:7 ]), image.key)
        _plot_scalebar      (plt.subplot(grid[ :2, 7  ]),
                             calib_result)

        _plot_histograms    (plt.subplot(grid[2:4,  :5]),
                             col_stats.histograms, col_stats.bins[:-1],
                             colors=colors, titles=titles,
                             **xtick_params)
        _plot_stats         (plt.subplot(grid[2:4, 5:7 ]),
                             image.stats, title="Structural and selected colour traits")

        _plot_boxplots      (plt.subplot(grid[4: ,  :5]),
                             channels, colors=colors, titles=titles, mask=mask,
                             **xtick_params)
        _plot_col_stats     (plt.subplot(grid[4: , 5: ]),
                             col_stats, colors=colors, title="")

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

    def save_img(self, orig_path: str, img: np.ndarray, *, subfolder: str):
        dest = self.new_path(orig_path, new_suffix=".jpg", subfolder=subfolder)
        plt.imsave(dest, img)
        return dest
