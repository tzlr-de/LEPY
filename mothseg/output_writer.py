import json
import csv

from pathlib import Path

from mothseg import PointsOfInterest
from mothseg import visualization as vis


class OutputWriter:

    def __init__(self, folder: str, *, store_to_csv: bool = True) -> None:
        self.root = Path(folder)

        self.root.mkdir(exist_ok=True, parents=True)
        self.csv = None
        if store_to_csv: 
            self._csv_file = open(self.root / "stats.csv", "w")
            self.csv = csv.writer(self._csv_file, delimiter="\t")
            self.header = [
                "Code",

                "median-intensity", "mean-intensity", "stddev-intensity",
                "median-saturation", "mean-saturation", "stddev-saturation", 
                "median-hue", "mean-hue", "stddev-hue", 

                "seg-absolute-size", "seg-relative-size", 

                "c-length", "c-area", "c-xmin", "c-xmax", "c-ymin", "c-ymax", 
                "c-area-calibrated", "width-calibrated", "height-calibrated",
                "calibration-length",

                "poi-orig_width", "poi-orig_height",
                "poi-center-x", "poi-center-y",
                "poi-outer_l-x", "poi-outer_l-y",
                "poi-outer_r-x", "poi-outer_r-y",
                "poi-inner_l-x", "poi-inner_l-y",
                "poi-inner_r-x", "poi-inner_r-y",
            ]

            self.csv.writerow(self.header)
            
    def __del__(self):
        if hasattr(self, "_csv_file"):
            self._csv_file.close()
    
    def new_path(self, impath: str, new_suffix: str):
        new_path = Path(impath).with_suffix(new_suffix).name
        return self.root / new_path

    def __call__(self, impath: str, stats: dict, *, missing_value: str = "") -> None:
        
        with open(self.new_path(impath, ".json"), "w") as f:
            json.dump(stats, f, indent=2)
        
        if self.csv is None:
            return
        
        row = [Path(impath).stem] + [stats.get(key, missing_value) for key in self.header[1:]]
        self.csv.writerow(row)
        self._csv_file.flush()
    
    def plot(self, impath: str, ims, contour, stats, pois: PointsOfInterest):
        dest = self.new_path(impath, ".png")
        fig = vis.plot(ims, contour, stats, pois=pois)
        fig.savefig(dest)


    def plot_interm(self, impath: str, im, interm, cal_length):
        dest = self.new_path(impath, "_interm.png")
        fig = vis.plot_interm(im, interm, cal_length)
        fig.savefig(dest)