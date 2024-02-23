import scalebar

from cvargparse import BaseParser
from cvargparse import Arg

def parse_args():
    parser = BaseParser([
        Arg("folder"),
        Arg.float("--rescale", default=None),

        Arg("--method", "-meth", 
            choices=["otsu", "grabcut", "grabcut+", "grabcut+otsu"],
            default="grabcut+otsu"
        ),

        Arg.flag("--calibration", "-calib"),
        Arg.flag("--pois", "-pois"),
        Arg.flag("--yes", "-y"),
        Arg("--calib_pos", choices=[pos.name.lower() for pos in scalebar.Position], default="bottom_left"),
        Arg.float("--calib_rel_height", default=0.3),
        Arg.float("--calib_rel_width", default=0.1),

        Arg.flag("--show_interm"),
    ])

    return parser.parse_args()