import scalebar

from matplotlib import pyplot as plt

from mothseg import PointsOfInterest
from mothseg import visualization as vis
from mothseg.image import ColorStats
from mothseg.image import Image
from mothseg.output_writer.base import BaseWriter

class Plotter(BaseWriter):

    def __init__(self, folder: str, *, plot_interm: bool) -> None:
        super().__init__(folder)
        self._plot_interm = plot_interm

    def plot(self, image: Image, ims, *,
             pois: PointsOfInterest, calib_result: scalebar.Result = None,
             col_stats: ColorStats = None):

        dest = self.new_path(image.rgb_path, ".png", subfolder="visualizations")
        fig = vis.plot(image, ims,
                       pois=pois, calib_result=calib_result,
                       col_stats=col_stats,
                       )
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
