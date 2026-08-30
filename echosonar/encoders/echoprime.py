"""The frozen EchoPrime clip encoder, view classifier and MIL aggregator.

This is the feature extractor behind ``clip_embeddings.h5`` and
``study_embeddings.h5``. Nothing here is trained by EchoSonar -- the weights are
EchoPrime's and are used strictly in inference mode.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
import torchvision

from ..constants import (
    CLIP_DIM,
    COARSE_VIEWS,
    ENCODER_WEIGHTS,
    NUM_MIL_SECTIONS,
    NUM_VIEWS,
    TOKEN_DIM,
    VIEW_CLASSIFIER_WEIGHTS,
)

LOCAL_MIL_WEIGHTS = Path(__file__).resolve().parents[2] / "assets" / "MIL_weights.csv"


def resolve_mil_weights(
    explicit: Union[str, Path, None], weights_dir: Path
) -> Path:
    """Locate ``MIL_weights.csv``.

    This is EchoPrime's file, not ours -- it ships in ``EchoPrime/assets/``,
    alongside the weights the user already had to download, so that's the
    copy this prefers. Also checks ``assets/MIL_weights.csv`` here in case a
    local copy was dropped in (see README), but that's not shipped by default.
    """
    if explicit:
        return Path(explicit)
    for parent in [weights_dir, *weights_dir.parents]:
        candidate = parent / "assets" / "MIL_weights.csv"
        if candidate.is_file():
            return candidate
    if LOCAL_MIL_WEIGHTS.is_file():
        return LOCAL_MIL_WEIGHTS
    raise FileNotFoundError(
        "MIL_weights.csv not found. Pass --mil-weights pointing at "
        "EchoPrime/assets/MIL_weights.csv."
    )


@dataclass
class ClipFeatures:
    """Per-clip output of the encoder."""

    embedding: torch.Tensor  # (512,) float32, on CPU
    view_onehot: torch.Tensor  # (11,) float32, on CPU
    view: str


@dataclass
class ClipTokens:
    """Per-clip spatiotemporal tokens -- the report-generation model's input."""

    tokens: torch.Tensor  # (393, 768) float32, on CPU
    view: str  # EchoPrime's own coarse (11-class) prediction
    fine_view: Optional[str] = None  # FineViewClassifier's prediction, if run


class EchoPrimeEncoder:
    """Loads the two frozen backbones and turns clips into features.

    Parameters
    ----------
    weights_dir:
        Directory holding ``echo_prime_encoder.pt`` and ``view_classifier.pt``
        from the EchoPrime release.
    device:
        Torch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    mil_weights_csv:
        Per-(section, view) MIL weights. Defaults to the copy in ``assets/``.
    """

    def __init__(
        self,
        weights_dir: Union[str, Path],
        device: Union[str, torch.device] = "cuda",
        mil_weights_csv: Union[str, Path, None] = None,
    ):
        import pandas as pd

        self.device = torch.device(device)
        weights_dir = Path(weights_dir)

        encoder_path = weights_dir / ENCODER_WEIGHTS
        view_path = weights_dir / VIEW_CLASSIFIER_WEIGHTS
        for path in (encoder_path, view_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing EchoPrime weight file: {path}\n"
                    "Download model_data from the EchoPrime release and point "
                    "--weights-dir at the directory containing it."
                )

        self.encoder = torchvision.models.video.mvit_v2_s()
        self.encoder.head[-1] = torch.nn.Linear(
            self.encoder.head[-1].in_features, CLIP_DIM
        )
        self.encoder.load_state_dict(
            torch.load(encoder_path, map_location="cpu", weights_only=True)
        )
        self._freeze(self.encoder)

        self.view_classifier = torchvision.models.convnext_base()
        self.view_classifier.classifier[-1] = torch.nn.Linear(
            self.view_classifier.classifier[-1].in_features, NUM_VIEWS
        )
        self.view_classifier.load_state_dict(
            torch.load(view_path, map_location="cpu", weights_only=True)
        )
        self._freeze(self.view_classifier)

        mil = pd.read_csv(resolve_mil_weights(mil_weights_csv, weights_dir))
        self.mil_sections: List[str] = list(mil["Section"])
        self.mil_weights = torch.tensor(mil.iloc[:, 1:].to_numpy(), dtype=torch.float)
        if self.mil_weights.shape != (NUM_MIL_SECTIONS, NUM_VIEWS):
            raise ValueError(
                f"MIL weights are {tuple(self.mil_weights.shape)}, "
                f"expected ({NUM_MIL_SECTIONS}, {NUM_VIEWS})"
            )
        if list(mil.columns[1:]) != COARSE_VIEWS:
            raise ValueError("MIL weight columns do not match COARSE_VIEWS order")

    def _freeze(self, module: torch.nn.Module) -> None:
        module.eval().to(self.device)
        for param in module.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def encode_clips(
        self, clips: Sequence[torch.Tensor], batch_size: int = 16
    ) -> List[ClipFeatures]:
        """Encode preprocessed clips.

        ``clips`` are ``(3, 16, 224, 224)`` tensors from
        :func:`echosonar.preprocess.preprocess_frames`. The view classifier runs
        on each clip's first frame, matching EchoPrime's own convention.
        """
        if not clips:
            return []

        embeddings, view_indices = [], []
        for start in range(0, len(clips), batch_size):
            batch = torch.stack(list(clips[start : start + batch_size])).to(self.device)
            embeddings.append(self.encoder(batch).cpu())
            logits = self.view_classifier(batch[:, :, 0, :, :])
            view_indices.append(logits.argmax(dim=1).cpu())

        embeddings = torch.cat(embeddings)
        view_indices = torch.cat(view_indices)
        view_onehots = F.one_hot(view_indices, NUM_VIEWS).float()

        return [
            ClipFeatures(
                embedding=embeddings[i],
                view_onehot=view_onehots[i],
                view=COARSE_VIEWS[int(view_indices[i])],
            )
            for i in range(len(clips))
        ]

    @torch.inference_mode()
    def encode_clip_tokens(
        self, clips: Sequence[torch.Tensor], batch_size: int = 16
    ) -> List[ClipTokens]:
        """Encode preprocessed clips into pre-pooling tokens, for report generation.

        Unlike :meth:`encode_clips`, this returns the ``(393, 768)`` token
        sequence out of the encoder's last transformer block instead of the
        final pooled embedding -- what the report-generation model was trained
        to consume. Captured via a forward hook, so the pooled head still runs
        but its output is unused here; this method and :meth:`encode_clips` can
        both be called on the same encoder instance.
        """
        if not clips:
            return []

        captured: dict = {}

        def _hook(module, inputs, output):
            # output is (tokens, thw_shape); tokens is (B, 393, 768)
            captured["tokens"] = output[0].cpu()

        handle = self.encoder.blocks[-1].register_forward_hook(_hook)
        try:
            tokens_out, view_indices = [], []
            for start in range(0, len(clips), batch_size):
                batch = torch.stack(list(clips[start : start + batch_size])).to(self.device)
                self.encoder(batch)
                tokens_out.append(captured["tokens"])
                logits = self.view_classifier(batch[:, :, 0, :, :])
                view_indices.append(logits.argmax(dim=1).cpu())
        finally:
            handle.remove()

        tokens = torch.cat(tokens_out)
        view_indices = torch.cat(view_indices)
        if tokens.shape[-1] != TOKEN_DIM:
            raise ValueError(f"expected {TOKEN_DIM}-d tokens, got {tokens.shape[-1]}")

        return [
            ClipTokens(tokens=tokens[i], view=COARSE_VIEWS[int(view_indices[i])])
            for i in range(len(clips))
        ]

    def encoded_study(self, clips: Sequence[ClipFeatures]) -> torch.Tensor:
        """Stack per-clip features into the ``(n_clips, 523)`` study matrix."""
        if not clips:
            raise ValueError("cannot build an encoded study from zero clips")
        return torch.cat(
            [
                torch.stack([c.embedding for c in clips]),
                torch.stack([c.view_onehot for c in clips]),
            ],
            dim=1,
        )

    def mil_embedding(self, encoded_study: torch.Tensor) -> torch.Tensor:
        """MIL-aggregate a study into ``(15, 512)`` -- one embedding per section.

        Each clip is weighted by how informative its view is for that section,
        summed, then L2-normalised per section.
        """
        clip_emb = encoded_study[:, :CLIP_DIM]
        views = encoded_study[:, CLIP_DIM:].argmax(dim=1)
        weights = self.mil_weights[:, views]  # (15, n_clips)
        per_section = (clip_emb.unsqueeze(0) * weights.unsqueeze(-1)).sum(dim=1)
        return F.normalize(per_section)


def resolve_device(device: Optional[str]) -> str:
    """Pick a sensible device, honouring CUDA_VISIBLE_DEVICES."""
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def set_worker_device(gpu_id: int) -> torch.device:
    """Bind a worker process to one GPU.

    ``torch.cuda.set_device`` is required as well as the device string: models
    that internally default to ``cuda:0`` otherwise raise device mismatches.
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


__all__ = [
    "ClipFeatures",
    "ClipTokens",
    "EchoPrimeEncoder",
    "resolve_device",
    "set_worker_device",
]
