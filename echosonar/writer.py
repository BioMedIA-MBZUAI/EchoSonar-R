"""HDF5 layout for the embedding files -- see the functions below for the schema.

Both files are keyed by id and written incrementally, so a run can be resumed
and clip embeddings are reused across studies that share a clip.
"""

import json
from typing import List, Sequence

import h5py
import torch

from .encoders.echoprime import ClipFeatures, ClipTokens


def write_clip(h5: h5py.File, clip_id: str, features: ClipFeatures) -> None:
    """Store one clip's features. No-op if already present."""
    if clip_id in h5:
        return
    group = h5.create_group(clip_id)
    group.create_dataset("embedding", data=features.embedding.numpy())
    group.create_dataset("view_onehot", data=features.view_onehot.numpy())
    group.attrs["view"] = features.view


def read_clip(h5: h5py.File, clip_id: str) -> ClipFeatures:
    """Load a previously cached clip."""
    group = h5[clip_id]
    return ClipFeatures(
        embedding=torch.tensor(group["embedding"][:]),
        view_onehot=torch.tensor(group["view_onehot"][:]),
        view=group.attrs["view"],
    )


def write_clip_tokens(h5: h5py.File, clip_id: str, tokens: ClipTokens) -> None:
    """Store one clip's spatiotemporal tokens. No-op if already present."""
    if clip_id in h5:
        return
    group = h5.create_group(clip_id)
    group.create_dataset(
        "tokens", data=tokens.tokens.numpy(), compression="gzip", compression_opts=4
    )
    group.attrs["view"] = tokens.view
    if tokens.fine_view is not None:
        group.attrs["fine_view"] = tokens.fine_view


def read_clip_tokens(h5: h5py.File, clip_id: str) -> ClipTokens:
    """Load a previously cached clip's tokens."""
    group = h5[clip_id]
    return ClipTokens(
        tokens=torch.tensor(group["tokens"][:]),
        view=group.attrs["view"],
        fine_view=group.attrs.get("fine_view"),
    )


def write_study(
    h5: h5py.File,
    study_id: str,
    encoded_study: torch.Tensor,
    mil_embedding: torch.Tensor,
    clip_ids: Sequence[str],
    views: Sequence[str],
) -> None:
    """Store one study's aggregated features. No-op if already present."""
    if study_id in h5:
        return
    group = h5.create_group(study_id)
    group.create_dataset("encoded_study", data=encoded_study.numpy())
    group.create_dataset("mil_embedding", data=mil_embedding.numpy())
    group.attrs["clip_uuids"] = json.dumps(list(clip_ids))
    group.attrs["views"] = json.dumps(list(views))


def set_provenance(h5: h5py.File, **fields) -> None:
    """Record how the file was produced, as root attributes."""
    for key, value in fields.items():
        if value is not None:
            h5.attrs[key] = str(value)


def existing_keys(path) -> List[str]:
    """Top-level keys already in an H5 file (empty list if it does not exist)."""
    try:
        with h5py.File(path, "r") as h5:
            return list(h5.keys())
    except (OSError, FileNotFoundError):
        return []
