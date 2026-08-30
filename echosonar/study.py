"""How a study on disk becomes a list of clips to encode.

Either point :func:`discover_studies` at a directory of echo frame folders, or
load a JSONL manifest -- see :func:`load_manifest` below for the format.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .io.video import is_clip_source, read_clip, read_frame_files


@dataclass
class Clip:
    """One echo video (a directory of PNG frames) within a study."""

    clip_id: str
    source: Optional[str] = None
    frame_paths: Optional[List[str]] = None
    view: Optional[str] = None  # if known up front; otherwise predicted

    def read(self) -> np.ndarray:
        """Load this clip as ``(n_frames, H, W, 3)`` RGB ``uint8``."""
        if self.frame_paths:
            return read_frame_files(self.frame_paths)
        if self.source:
            return read_clip(self.source)
        raise ValueError(f"clip {self.clip_id} has neither source nor frame_paths")


@dataclass
class Study:
    """A set of clips acquired in one echo exam."""

    study_id: str
    clips: List[Clip] = field(default_factory=list)


def _clip_from_dict(entry: dict, index: int) -> Clip:
    clip_id = entry.get("clip_id") or entry.get("dicom_uuid") or entry.get("dicom_id")
    if not clip_id:
        clip_id = f"clip_{index:03d}"
    return Clip(
        clip_id=str(clip_id),
        source=entry.get("source") or entry.get("path"),
        frame_paths=entry.get("frame_paths"),
        view=entry.get("view") or entry.get("view_name"),
    )


def load_manifest(path: Union[str, Path]) -> List[Study]:
    """Read studies from a JSONL manifest."""
    studies = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            study_id = record.get("study_id") or record.get("study_uuid")
            if not study_id:
                raise ValueError(f"{path}:{line_no} has no study_id")
            entries = record.get("clips") or record.get("dicoms") or []
            if not entries:
                raise ValueError(f"{path}:{line_no} ({study_id}) lists no clips")
            studies.append(
                Study(
                    study_id=str(study_id),
                    clips=[_clip_from_dict(e, i) for i, e in enumerate(entries)],
                )
            )
    return studies


def load_study_dir(path: Union[str, Path]) -> Study:
    """One directory of echo videos -> one :class:`Study`.

    Each sub-directory of extracted PNG frames inside becomes a clip, named
    after the directory name.
    """
    path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(path)

    clips = [
        Clip(clip_id=entry.name if entry.is_dir() else entry.stem, source=str(entry))
        for entry in sorted(path.iterdir())
        if is_clip_source(entry)
    ]
    if not clips:
        raise ValueError(
            f"no echo clips found in {path} -- expected sub-directories of "
            "extracted PNG frames"
        )
    return Study(study_id=path.name, clips=clips)


def discover_studies(root: Union[str, Path], multi: bool = False) -> List[Study]:
    """Find studies under ``root``.

    ``multi=False`` treats ``root`` itself as a single study; ``multi=True``
    treats each immediate sub-directory as its own study.
    """
    root = Path(root)
    if not multi:
        return [load_study_dir(root)]

    studies = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            studies.append(load_study_dir(entry))
        except ValueError:
            continue  # not a study directory
    if not studies:
        raise ValueError(f"no study sub-directories found under {root}")
    return studies
