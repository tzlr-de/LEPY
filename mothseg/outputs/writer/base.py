
from pathlib import Path

class BaseWriter:
    def __init__(self, folder: str) -> None:
        if folder is None:
            self.root = None
        else:
            self.root = Path(folder)
            self.root.mkdir(exist_ok=True, parents=True)

    def new_path(self, impath: str, new_suffix: str, *, subfolder: str = None):
        if self.root is None:
            return None

        new_path = Path(impath).with_suffix(new_suffix).name
        if subfolder is None:
            return self.root / new_path
        else:
            subpath = self.root / subfolder
            subpath.mkdir(exist_ok=True, parents=True)
            return subpath / new_path
