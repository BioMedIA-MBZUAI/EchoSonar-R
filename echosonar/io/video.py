"""Load one echo clip as RGB frames from a directory of PNG frames.

Supported: directories of extracted PNG frames only.
"""

import re
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np

IMAGE_EXTS = {".png"}

_TRAILING_INT = re.compile(r"(\d+)\s*$")


def _natural_key(path: Path):
    """Sort ``frame_2.png`` before ``frame_10.png``.

    Lexicographic ordering would interleave them, silently shuffling the clip.
    """
    match = _TRAILING_INT.search(path.stem)
    return (0, int(match.group(1)), "") if match else (1, 0, path.stem)


def list_frame_files(directory: Union[str, Path]) -> List[Path]:
    """Frame images inside a directory, in acquisition order."""
    directory = Path(directory)
    files = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=_natural_key)


def read_frame_dir(directory: Union[str, Path]) -> np.ndarray:
    """Directory of PNG frames -> ``(n_frames, H, W, 3)`` RGB ``uint8``."""
    files = list_frame_files(directory)
    if not files:
        raise ValueError(f"no PNG frames in {directory}")
    return read_frame_files(files)


def read_frame_files(paths) -> np.ndarray:
    """Explicit list of image paths -> ``(n, H, W, 3)`` RGB ``uint8``."""
    frames = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"failed to read {path}")
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def read_clip(source: Union[str, Path]) -> np.ndarray:
    """Read a clip source as ``(n_frames, H, W, 3)`` RGB ``uint8``.

    ``source`` must be a directory of PNG frames.
    """
    source = Path(source)
    if source.is_dir():
        return read_frame_dir(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() in IMAGE_EXTS:
        raise ValueError(
            f"{source} is a single image; point at the directory containing the "
            "clip's frames instead"
        )
    raise ValueError(f"unsupported clip source: {source} (only PNG frame directories are supported)")


def is_clip_source(path: Union[str, Path]) -> bool:
    """True if ``path`` looks like one clip (a directory of PNG frames)."""
    path = Path(path)
    if path.is_dir():
        return bool(list_frame_files(path))
    return False
