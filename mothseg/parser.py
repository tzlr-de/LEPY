from cvargparse import BaseParser
from cvargparse import Arg

def parse_args():
    parser = BaseParser([
        Arg("folder"),
        Arg("config"),
        Arg.float("--rescale", default=None),

        Arg("--output", "-o"),

        Arg.flag("--yes", "-y"),
        
        Arg.flag("--show_interm"),
        Arg.flag("--use_timestamp", "-use_ts"),
    ])

    return parser.parse_args()