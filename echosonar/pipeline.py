"""Study -> clip embeddings -> study embeddings.

This is the whole feature-extraction path in one place; ``scripts/
extract_echoprime_embeddings.py`` is a thin CLI over :func:`extract_studies`.
"""

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import h5py
import torch

from . import writer
from .encoders.echoprime import ClipFeatures, EchoPrimeEncoder, resolve_device
from .encoders.view_classifier import FineViewClassifier
from .preprocess import preprocess_frames, sample_frame_indices
from .study import Study

logger = logging.getLogger(__name__)


def encode_clip_features(
    encoder: EchoPrimeEncoder,
    study: Study,
    clip_h5: h5py.File,
    frame_sampling: str = "first32",
    batch_size: int = 16,
) -> List[ClipFeatures]:
    """Features for every clip in a study, in study order.

    Clips already present in ``clip_h5`` are read back rather than re-encoded,
    which is what makes the whole extraction resumable.
    """
    cached, pending, pending_clips = {}, [], []

    for clip in study.clips:
        if clip.clip_id in clip_h5:
            cached[clip.clip_id] = writer.read_clip(clip_h5, clip.clip_id)
            continue
        frames = clip.read()
        indices = sample_frame_indices(len(frames), strategy=frame_sampling)
        pending.append(preprocess_frames(frames[indices]))
        pending_clips.append(clip)

    for clip, features in zip(
        pending_clips, encoder.encode_clips(pending, batch_size=batch_size)
    ):
        writer.write_clip(clip_h5, clip.clip_id, features)
        cached[clip.clip_id] = features

    return [cached[clip.clip_id] for clip in study.clips]


def encode_study(
    encoder: EchoPrimeEncoder,
    study: Study,
    clip_h5: h5py.File,
    study_h5: h5py.File,
    frame_sampling: str = "first32",
    batch_size: int = 16,
) -> None:
    """Encode one study and append it to both open H5 files."""
    features = encode_clip_features(
        encoder,
        study,
        clip_h5,
        frame_sampling=frame_sampling,
        batch_size=batch_size,
    )
    encoded_study = encoder.encoded_study(features)
    writer.write_study(
        study_h5,
        study.study_id,
        encoded_study=encoded_study,
        mil_embedding=encoder.mil_embedding(encoded_study),
        clip_ids=[clip.clip_id for clip in study.clips],
        views=[f.view for f in features],
    )


def extract_studies(
    studies: Sequence[Study],
    weights_dir: Union[str, Path],
    clip_h5_path: Union[str, Path],
    study_h5_path: Union[str, Path],
    device: Optional[str] = None,
    frame_sampling: str = "first32",
    batch_size: int = 16,
    resume: bool = True,
    progress: bool = True,
) -> dict:
    """Encode studies into ``clip_h5_path`` and ``study_h5_path``.

    Returns a summary dict with counts and the ids of any studies that failed.
    Failures are logged and skipped -- one unreadable clip should not abandon a
    long run.
    """
    encoder = EchoPrimeEncoder(weights_dir, device=resolve_device(device))

    Path(clip_h5_path).parent.mkdir(parents=True, exist_ok=True)
    Path(study_h5_path).parent.mkdir(parents=True, exist_ok=True)

    done = set(writer.existing_keys(study_h5_path)) if resume else set()
    todo = [s for s in studies if s.study_id not in done]
    if done:
        logger.info("resuming: %d studies already done, %d to go", len(done), len(todo))

    iterator = todo
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(todo, desc="studies", unit="study")
        except ImportError:
            pass

    failed = []
    mode = "a" if resume else "w"
    with h5py.File(clip_h5_path, mode) as clip_h5, h5py.File(
        study_h5_path, mode
    ) as study_h5:
        writer.set_provenance(
            clip_h5, frame_sampling=frame_sampling, extractor="echoprime"
        )
        writer.set_provenance(
            study_h5, frame_sampling=frame_sampling, extractor="echoprime"
        )

        for i, study in enumerate(iterator, 1):
            try:
                encode_study(
                    encoder,
                    study,
                    clip_h5,
                    study_h5,
                    frame_sampling=frame_sampling,
                    batch_size=batch_size,
                )
            except Exception as exc:
                logger.error("study %s failed: %s", study.study_id, exc)
                failed.append(study.study_id)
            if i % 50 == 0:
                clip_h5.flush()
                study_h5.flush()

        n_clips = len(clip_h5.keys())
        n_studies = len(study_h5.keys())

    return {
        "studies_written": n_studies,
        "clips_written": n_clips,
        "skipped_existing": len(done),
        "failed": failed,
    }


def extract_clip_tokens(
    studies: Sequence[Study],
    weights_dir: Union[str, Path],
    tokens_h5_path: Union[str, Path],
    view_classifier_checkpoint: Union[str, Path, None] = None,
    device: Optional[str] = None,
    frame_sampling: str = "first32",
    batch_size: int = 16,
    resume: bool = True,
    progress: bool = True,
) -> dict:
    """Encode every clip's spatiotemporal tokens into ``tokens_h5_path``.

    This is the report-generation model's vision input, distinct from
    ``clip_embeddings.h5`` (the pooled embedding used for MIL). See
    :meth:`EchoPrimeEncoder.encode_clip_tokens`.

    ``view_classifier_checkpoint`` -- typically
    ``checkpoints/view_classifier.pt`` -- additionally predicts each
    clip's *fine* view label (see :class:`FineViewClassifier`), the vocabulary
    the report model was actually trained on. Without it, only EchoPrime's own
    coarser view prediction is stored.
    """
    resolved_device = resolve_device(device)
    encoder = EchoPrimeEncoder(weights_dir, device=resolved_device)
    fine_view_classifier = (
        FineViewClassifier(view_classifier_checkpoint, device=resolved_device)
        if view_classifier_checkpoint
        else None
    )

    Path(tokens_h5_path).parent.mkdir(parents=True, exist_ok=True)

    clips = [clip for study in studies for clip in study.clips]
    done = set(writer.existing_keys(tokens_h5_path)) if resume else set()
    todo = [clip for clip in clips if clip.clip_id not in done]
    if done:
        logger.info("resuming: %d clips already done, %d to go", len(done), len(todo))

    iterator = todo
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(todo, desc="clips", unit="clip")
        except ImportError:
            pass

    def _flush(pending, pending_raw, pending_clips, tokens_h5):
        for c, raw, tokens in zip(
            pending_clips, pending_raw, encoder.encode_clip_tokens(pending, batch_size=batch_size)
        ):
            if fine_view_classifier is not None:
                tokens.fine_view = fine_view_classifier.predict_clip(raw)
            writer.write_clip_tokens(tokens_h5, c.clip_id, tokens)

    failed = []
    mode = "a" if resume else "w"
    with h5py.File(tokens_h5_path, mode) as tokens_h5:
        writer.set_provenance(
            tokens_h5, frame_sampling=frame_sampling, extractor="echoprime_tokens"
        )

        pending, pending_raw, pending_clips = [], [], []
        for clip in iterator:
            try:
                frames = clip.read()
                indices = sample_frame_indices(len(frames), strategy=frame_sampling)
                sampled = frames[indices]
                pending.append(preprocess_frames(sampled))
                pending_raw.append(sampled)
                pending_clips.append(clip)
            except Exception as exc:
                logger.error("clip %s failed: %s", clip.clip_id, exc)
                failed.append(clip.clip_id)

            if len(pending) >= batch_size:
                _flush(pending, pending_raw, pending_clips, tokens_h5)
                tokens_h5.flush()
                pending, pending_raw, pending_clips = [], [], []

        _flush(pending, pending_raw, pending_clips, tokens_h5)

        n_clips = len(tokens_h5.keys())

    return {
        "clips_written": n_clips,
        "skipped_existing": len(done),
        "failed": failed,
    }


def encode_single_study(
    study: Study,
    weights_dir: Union[str, Path],
    device: Optional[str] = None,
    frame_sampling: str = "first32",
    batch_size: int = 16,
):
    """In-memory variant: returns ``(encoded_study, mil_embedding, views)``.

    Convenient when a downstream model wants the features directly and there is
    no reason to touch disk.
    """
    encoder = EchoPrimeEncoder(weights_dir, device=resolve_device(device))
    clips = []
    for clip in study.clips:
        frames = clip.read()
        indices = sample_frame_indices(len(frames), strategy=frame_sampling)
        clips.append(preprocess_frames(frames[indices]))

    features = encoder.encode_clips(clips, batch_size=batch_size)
    encoded_study = encoder.encoded_study(features)
    mil = encoder.mil_embedding(encoded_study)
    return encoded_study, mil, [f.view for f in features]
