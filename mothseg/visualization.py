import cv2
import numpy as np
import matplotlib.pyplot as plt
import typing as T

from mothseg import PointsOfInterest

def plot(ims, contour, stats, pois: T.Optional[PointsOfInterest] = None):

    fig, axs = plt.subplots(nrows=1, ncols=len(ims), figsize=(16,9), squeeze=False)

    for _, _im in enumerate(ims):
        ax = axs[np.unravel_index(_, axs.shape)]
        
        ax.imshow(_im)
        ax.plot(contour[:, 0], contour[:, 1], linewidth=2, alpha=0.6)
        
        rx = [stats['c-xmin'], stats['c-xmax'], stats['c-xmax'], stats['c-xmin'], stats['c-xmin']]
        ry = [stats['c-ymin'], stats['c-ymin'], stats['c-ymax'], stats['c-ymax'], stats['c-ymin']]
        ax.plot(rx, ry, 'm--', linewidth=0.5)
        ax.arrow(stats['c-xmin'], stats['c-ymax'], stats['c-xmax'] - stats['c-xmin'], 0, length_includes_head=True,
                            width=3)
        ax.arrow(stats['c-xmax'], stats['c-ymax'], -stats['c-xmax'] + stats['c-xmin'], 0, length_includes_head=True,
                        width=3)

        if 'calibration-length' not in stats:
            lengthx = '{:.0f} pixels'.format(stats['c-xmax'] - stats['c-xmin'])
        else:
            lengthx = '{:.2f} mm'.format(stats['width-calibrated'])

        ax.text(0.5 * (stats['c-xmax'] + stats['c-xmin']), stats['c-ymax'] + 20,
                 lengthx,
                 horizontalalignment='center', verticalalignment='top', fontsize=18)

        if pois is not None:
            for name, poi in pois:
                ax.scatter(poi.col, poi.row, color='r')
    plt.show()
    plt.close()


def plot_interm(im, interm, px_per_mm=None):
    H, W, *C = im.shape
    
    fig = plt.figure(figsize=(16,9))
    grid = plt.GridSpec(3, 2)

    ax = plt.subplot(grid[:1, :])
    ax.axis("off")
    ax.imshow(im)
    ax.set_title("Input image")

    crop = interm["crop"]
    init_corners = interm["detected_corners"]
    mask = interm["filter_mask"]
    angle = interm["rectification_angle"]
    corners = interm["final_corners"]

    ax = plt.subplot(grid[-2:, 0])
    ax.axis("off")
    ax.imshow(crop)
    ax.set_title("Original crop")


    ys, xs = init_corners[mask].transpose(1, 0)
    ax.scatter(xs, ys, marker=".", c="red", label="used")
    
    if px_per_mm is not None:
        for cx, cy in zip(xs, ys):
            ax.arrow(cx, cy, px_per_mm, 0, width=1, length_includes_head=True)
    
    ys, xs = init_corners[~mask].transpose(1, 0)
    ax.scatter(xs, ys, marker=".", c="blue", alpha=0.7, label="rejected")
    ax.legend(loc="upper right")

    ax = plt.subplot(grid[-2:, 1])
    rot_mat = cv2.getRotationMatrix2D([0, 0], angle, 1.0)
    H, W, *C = crop.shape
    new_crop = cv2.warpAffine(crop, rot_mat, (W, H))

    ax.imshow(new_crop)
    ax.axis("off")
    ax.set_title("Rectified crop")
    ys, xs = corners.transpose(1, 0)
    ax.scatter(xs, ys, marker=".", c="red")

    if px_per_mm is None:
        fig.suptitle("Estimation Failed!")
    else:
        size = W / px_per_mm, H / px_per_mm
        fig.suptitle(" | ".join(
            [
                f"{px_per_mm:.2f} px/mm",
                f"Image size: {size[0]:.2f} x {size[1]:.2f}mm"
            ]))

    plt.tight_layout()
    plt.show()
    plt.close()