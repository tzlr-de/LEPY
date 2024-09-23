import numpy as np
import matplotlib.pyplot as plt
import typing as T
import scalebar
import cv2

from mothseg.poi import PointsOfInterest
from mothseg.outputs import OUTPUTS as OUTS
from mothseg.image import ColorStats
from mothseg.image import Image

def plot(image: Image, ims, *,
         pois: T.Optional[PointsOfInterest] = None,
         calib_result: T.Optional[scalebar.Result] = None,
         col_stats: T.Optional[ColorStats] = None,
         colored_rgb: bool = False) -> plt.Figure:

    im, uv, mask = image.rgb_im, image.uv_im, image.mask

    gray = image.gray_im
    intensity_img = image.intensity_im

    fig = plt.figure(figsize=(16, 9))
    grid = plt.GridSpec(5, 7, figure=fig)

    saturation = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)[:, :, 1]
    saturation -= saturation.min()
    saturation = (saturation / saturation.max() * 255).astype(np.uint8)

    ims = [im, uv, gray, saturation, mask]
    titles = ["RGB image", "UV image", "B/W image", "Saturation",  "Mask"]
    for i, (chan, title) in enumerate(zip(ims, titles)):
        ax = plt.subplot(grid[0, i])
        if chan.ndim == 2:
            ax.imshow(chan, cmap="gray")
        else:
            ax.imshow(chan)
        ax.set_title(title)
        ax.axis("off")

    colors = ["red", "green", "blue", "purple", "black"]
    titles = ["Red channel", "Green channel", "Blue channel", "UV channel", "B/W + UV image"]

    R, G, B = im.transpose(2, 0, 1)
    channels = [R, G, B, uv, intensity_img]
    # cmaps = ["Reds", "Greens", "Blues", "Purples", "Greys"]
    # cmaps = ["Reds_r", "Greens_r", "Blues_r", "Purples_r", "Greys_r"]
    cmaps = ["gray", "gray", "gray", "gray", "gray"]
    boxplot_ax = plt.subplot(grid[4, :5])
    tab_ax = plt.subplot(grid[4, 5])
    hist_ax = plt.subplot(grid[2:4, :5])

    rows = []
    for i, (stat, col, chan, title, cmap) in enumerate(zip(col_stats, colors, channels, titles, cmaps)):
        hist, q25, q75, median, iqr = stat

        hist_ax.plot(col_stats.bins[:-1], hist, color=col, label=title)
        hist_ax.fill_between(col_stats.bins[:-1], hist, color=col, alpha=0.3)

        ax = plt.subplot(grid[1, i])

        if i < 3 and colored_rgb:
            _chan = np.full_like(im, 0, dtype=np.uint8)
            _chan[..., i] = chan
            ax.imshow(_chan * mask[..., None])
        else:
            ax.imshow(chan * mask, cmap=cmap)

        _title_obj = ax.set_title(f"{title}\n(masked)")
        _title_obj.set_color(col)
        ax.axis("off")

        # ax = plt.subplot(grid[2, i])
        boxplot_ax.boxplot(chan[mask != 0], patch_artist=True,
                           vert=False,
                           widths=0.7,
                           boxprops=dict(facecolor=col),
                           medianprops=dict(color='yellow'),
                           positions=[len(channels) - i],
                           showfliers=False)

        rows.append([title, int(q25), int(median), int(q75), int(iqr)])
        # boxplot_ax.set_title(f"Q25: {q25:.0f}, Q75: {q75:.0f}\nMedian: {median:.0f}, IQR: {iqr:.0f}")
        # boxplot_ax.axis("off")

    tab_ax.axis("off")
    tab_ax.axis("tight")
    tab  = tab_ax.table(cellText=[row[1:] for row in rows],
                 colLabels=[
                    #  "Channel",
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

    # print(tabulate(rows, headers=["Channel", "Q25", "Median", "Q75", "IQR"], tablefmt="fancy_grid"))
    # boxplot_ax.grid()
    boxplot_ax.set_xlim(-1, 256)
    boxplot_ax.set_xticks(np.linspace(0, 256, 5))
    boxplot_ax.set_xticks(np.linspace(0, 256, 5*5), minor=True)
    boxplot_ax.set_yticklabels(titles)

    # hist_ax.grid()
    hist_ax.set_xlim(-1, 256)
    hist_ax.set_xticks(np.linspace(0, 256, 5))
    hist_ax.set_xticks(np.linspace(0, 256, 5*5), minor=True)
    hist_ax.legend()

    plt.tight_layout()
    return fig




    stats = image.stats
    contour = image.contour

    nrows = len(ims)
    ncols = 1
    if calib_result is not None:
        ncols += 1

    if calib_result.match is not None:
        ncols += 1

    fig = plt.figure(figsize=(16,9))
    spec = plt.GridSpec(nrows=nrows, ncols=ncols, figure=fig)
    #fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16,9), squeeze=False)

    for _, _im in enumerate(ims):
        ax = plt.subplot(spec[_, 0]) #axs[np.unravel_index(_, axs.shape)]

        ax.imshow(_im)
        ax.plot(contour[:, 0], contour[:, 1], linewidth=2, alpha=0.6)
        x0, y0, x1, y1 = stats[OUTS.contour.xmin], stats[OUTS.contour.ymin], stats[OUTS.contour.xmax], stats[OUTS.contour.ymax]

        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=None, edgecolor='m', linewidth=1))
        ax.annotate("", xy=(x0, y1), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='<->'))

        if OUTS.calibration.length not in stats:
            lengthx = '{:.0f} pixels'.format(x1 - x0)
        else:
            lengthx = '{:.2f} mm'.format(stats[OUTS.contour.width_calibrated])

        ax.text(0.5 * (x1 + x0), y1 + 20,
                 lengthx,
                 horizontalalignment='center', verticalalignment='top', fontsize=18)

        if pois is not None:
            for name, poi in pois:
                ax.scatter(poi.col, poi.row, color='r')

            for key, p0, p1 in pois.named_distances:
                dist = stats[key]
                unit = "px" if OUTS.calibration.length not in stats else "mm"
                ax.annotate("", xy=(p0.col, p0.row), xytext=(p1.col, p1.row),
                            arrowprops=dict(arrowstyle='<->'))
                ax.text(x=(p0.col + p1.col)/2, y=(p0.row + p1.row)/2 + 15,
                        s=f"{dist:.2f} {unit}",
                        horizontalalignment='center',
                        verticalalignment='top',
                        fontsize=10,
                )

    if calib_result is not None:

        ax = plt.subplot(spec[:, 1])
        scalebar_crop = calib_result.scalebar
        ax.imshow(scalebar_crop, cmap=plt.cm.gray)
        ax.set_title(f"Detected scalebar: \n{calib_result.scale:.2f} px/mm")

        if calib_result.distances is not None:
            corners = calib_result.distances.corners
            mask = np.zeros(len(corners), dtype=bool)
            selected_corners, pairs = calib_result.best_corners()
            mask[selected_corners] = True

            ys, xs = corners[mask].transpose(1, 0)
            ax.scatter(xs, ys, marker="o", c="blue")

            ys, xs = corners[~mask].transpose(1, 0)
            ax.scatter(xs, ys, marker="o", c="red", alpha=0.5)

        if calib_result.match is not None:
            ax = plt.subplot(spec[0, 2])
            templ = calib_result.template
            ax.imshow(templ, cmap=plt.cm.gray)
            ax.set_title("Artifitial scalebar template")

            ax = plt.subplot(spec[1:, 2])
            match = calib_result.match
            ax.imshow(_im)
            ax.imshow(match, cmap=plt.cm.gray, alpha=0.5)
            ax.set_title("Template matches")

    plt.tight_layout()
    return fig


def imshow(ims):

    if len(ims) <= 3:
        nrows, ncols = 1, len(ims)
    else:
        nrows = int(np.ceil(np.sqrt(len(ims))))
        ncols = int(np.ceil( len(ims) / nrows))

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows,
                            figsize=(16,9), squeeze=False)
    for i, (title, im, cmap) in enumerate(ims):
        ax = axs[np.unravel_index(i, axs.shape)]

        if isinstance(im, (list, tuple)):
            alpha = 1 / len([_im for _im in im if _im is not None])
            for _im, _cm in zip(im, cmap):
                if _im is not None:
                    ax.imshow(_im, cmap=_cm, alpha=alpha)
        else:
            ax.imshow(im, cmap=cmap)
        ax.set_title(title)

    for _ in range(i+1, nrows*ncols):
        ax = axs[np.unravel_index(_, axs.shape)]
        ax.axis("off")

    return fig, axs


def plot_interm(result: scalebar.Result):
    images = result.images
    im = result.images.original

    ROI = scalebar.utils.hide_non_roi(images.binary, result.roi_fraction, 127,
                                      location=result.scalebar_location)
    scalebar_crop = result.position.crop(images.equalized)
    match_crop = None
    if result.match is not None:
        match_crop = result.position.crop(result.match)
    px_per_mm = result.scale


    fig, axs = imshow([
        ("Input image", im, plt.cm.gray),
        ("B/W image", images.gray, plt.cm.gray),
        ("B/W image equalized", images.equalized, plt.cm.gray),

        ("Binarized", images.binary, plt.cm.gray),
        ("ROI to be masked", ROI, plt.cm.gray),
        ("Masked", images.masked, plt.cm.gray),

        # ("Template", , plt.cm.gray),
        ("Template Matches", (images.binary, result.match), (plt.cm.gray, plt.cm.viridis)),

        ("Cropped template matches", (scalebar_crop, match_crop), (plt.cm.gray, plt.cm.viridis)),
        (f"Scalebar | {px_per_mm} px/mm", scalebar_crop, plt.cm.gray),
    ])
    ax = axs[np.unravel_index(8, axs.shape)]
    ys, xs = result.distances.corners.transpose(1, 0)
    ax.scatter(xs, ys, marker=".", c="red", alpha=0.3)


    if px_per_mm is None:
        fig.suptitle("Estimation Failed!")

    else:
        W, H = result.image_size
        size = W / px_per_mm, H / px_per_mm
        fig.suptitle(" | ".join(
            [
                f"{px_per_mm:.2f} px/mm",
                f"Image size: {size[0]:.2f} x {size[1]:.2f}mm"
            ])
        )

    plt.tight_layout()
    return fig
