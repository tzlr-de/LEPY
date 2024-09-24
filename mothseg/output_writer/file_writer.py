import json
import csv
import shutil
import datetime as dt

from pathlib import Path

from mothseg.outputs import OUTPUTS
from mothseg.output_writer.base import BaseWriter

class OutputWriter(BaseWriter):

    def __init__(self, folder: str, *, store_to_csv: bool = True, config = None) -> None:
        super().__init__(folder)
        shutil.copy(config, self.root / Path(config).name)

        self._csv_file = None
        if store_to_csv:
            self._csv_file = self.root / "stats.csv"
            with open(self._csv_file, "w"):
                pass # just clear the file

            self.header = [
                "Code",

                OUTPUTS.image.width, OUTPUTS.image.height,
                OUTPUTS.intensity.median, OUTPUTS.intensity.mean, OUTPUTS.intensity.stddev,
                OUTPUTS.saturation.median, OUTPUTS.saturation.mean, OUTPUTS.saturation.stddev,
                OUTPUTS.hue.median, OUTPUTS.hue.mean, OUTPUTS.hue.stddev,

                OUTPUTS.contour.length, OUTPUTS.contour.area,
                OUTPUTS.contour.xmin, OUTPUTS.contour.xmax,
                OUTPUTS.contour.ymin, OUTPUTS.contour.ymax,
                OUTPUTS.contour.area_calibrated,
                OUTPUTS.contour.width_calibrated,
                OUTPUTS.contour.height_calibrated,

                OUTPUTS.calibration.length,
                OUTPUTS.calibration.pos.x, OUTPUTS.calibration.pos.y,
                OUTPUTS.calibration.pos.w, OUTPUTS.calibration.pos.h,

                OUTPUTS.poi.area.body, OUTPUTS.poi.area.wing_l, OUTPUTS.poi.area.wing_r,

                OUTPUTS.poi.dist.inner_outer_l, OUTPUTS.poi.dist.inner_outer_r,
                OUTPUTS.poi.dist.inner, OUTPUTS.poi.dist.body,

                OUTPUTS.poi.orig_width, OUTPUTS.poi.orig_height,
                OUTPUTS.poi.center.x, OUTPUTS.poi.center.y,
                OUTPUTS.poi.body_top.x, OUTPUTS.poi.body_top.y,
                OUTPUTS.poi.body_bot.x, OUTPUTS.poi.body_bot.y,
                OUTPUTS.poi.outer_l.x, OUTPUTS.poi.outer_l.y, OUTPUTS.poi.outer_r.x, OUTPUTS.poi.outer_r.y,
                OUTPUTS.poi.inner_top_l.x, OUTPUTS.poi.inner_top_l.y, OUTPUTS.poi.inner_top_r.x, OUTPUTS.poi.inner_top_r.y,
                OUTPUTS.poi.inner_bot_l.x, OUTPUTS.poi.inner_bot_l.y, OUTPUTS.poi.inner_bot_r.x, OUTPUTS.poi.inner_bot_r.y,
            ]

            self.write_csv_row(self.header)

        self._err_file = self.root / "errors.log"
        with open(self._err_file, "w"):
            pass # just clear the file

    def write_csv_row(self, row, *, delimiter="\t"):
        with open(self._csv_file, "a") as f:
            csv.writer(f, delimiter=delimiter).writerow(row)
            f.flush()

    def __call__(self, impath: str, stats: dict, *, missing_value: str = "") -> None:

        with open(self.new_path(impath, ".json", subfolder="json"), "w") as f:
            json.dump(stats, f, indent=2)

        if self._csv_file is None:
            return

        row = [Path(impath).stem] + [stats.get(key, missing_value) for key in self.header[1:]]
        self.write_csv_row(row)

    def log_fail(self, impath: str, err: Exception):
        if not hasattr(self, "_err_file"):
            return
        now = dt.datetime.now()

        with open(self._err_file, "a") as f:
            msg = f"[{now:%Y-%m-%d %H:%M:%S}] Failed to process \"{impath}\". Reason ({type(err).__name__}): {str(err)}"
            f.write(f"{msg}\n")
            print(msg)
