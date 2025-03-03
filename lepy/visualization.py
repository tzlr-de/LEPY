import numpy as np
import matplotlib.pyplot as plt
import scalebar

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
