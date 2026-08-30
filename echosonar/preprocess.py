"""Frame sampling and pixel preprocessing for the EchoPrime clip encoder.

A clip of arbitrary length becomes a ``(3, 16, 224, 224)`` normalised tensor.
The two steps are kept separate because frame *sampling* is a dataset choice
(and differs between our internal and public pipelines) while the pixel
transform is fixed by the EchoPrime weights.
"""

from typing import List, Sequence

import numpy as np
import torch

from .constants import ENCODER_MEAN, ENCODER_STD, NUM_FRAMES, VIDEO_SIZE
from .third_party.echoprime import crop_and_scale

FRAME_SAMPLING_STRATEGIES = ("first32", "uniform")


def sample_frame_indices(
    n_available: int,
    num_frames: int = NUM_FRAMES,
    strategy: str = "first32",
) -> List[int]:
    """Pick ``num_frames`` frame indices out of a clip of ``n_available`` frames.

    ``first32`` (default) reproduces the rule used to build the released
    embeddings: short clips are padded by repeating the last frame, clips up to
    2x the target are truncated, and longer clips are strided by 2 over the
    first ``2 * num_frames`` frames -- i.e. roughly the first cardiac cycle.

    ``uniform`` spreads the samples over the whole clip instead. It sees more of
    the acquisition but does **not** match the released embeddings; use it only
    when regenerating a whole dataset consistently.
    """
    if n_available < 1:
        raise ValueError("clip has no frames")

    if strategy == "first32":
        if n_available <= num_frames:
            pad = [n_available - 1] * (num_frames - n_available)
            return list(range(n_available)) + pad
        if n_available <= 2 * num_frames:
            return list(range(num_frames))
        return list(range(0, 2 * num_frames, 2))

    if strategy == "uniform":
        # linspace to 2x the target over the whole clip, then take every other
        # frame -- the sampling used for the public-dataset evaluations.
        dense = np.linspace(0, n_available - 1, 2 * num_frames).round().astype(int)
        return dense[::2].tolist()

    raise ValueError(
        f"unknown frame sampling strategy {strategy!r}; "
        f"expected one of {FRAME_SAMPLING_STRATEGIES}"
    )


def preprocess_frames(frames: Sequence[np.ndarray]) -> torch.Tensor:
    """``num_frames`` RGB uint8 frames -> normalised ``(3, T, 224, 224)`` tensor.

    Frames must be ``uint8``: cubic resize on uint8 clips and rounds, so
    converting to float beforehand would shift the embeddings.
    """
    stack = np.zeros((len(frames), VIDEO_SIZE, VIDEO_SIZE, 3), dtype=np.float32)
    for i, frame in enumerate(frames):
        if frame.dtype != np.uint8:
            raise TypeError(f"frame {i} is {frame.dtype}, expected uint8")
        stack[i] = crop_and_scale(frame, res=(VIDEO_SIZE, VIDEO_SIZE))

    x = torch.as_tensor(stack, dtype=torch.float).permute(3, 0, 1, 2)
    x.sub_(ENCODER_MEAN).div_(ENCODER_STD)
    return x
