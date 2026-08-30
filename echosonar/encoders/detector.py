"""RT-DETR cardiac structure detector: boxes plus decoder query embeddings.

For each frame the detector returns bounding boxes over 7 cardiac structures,
and -- via a forward hook on the decoder -- the 256-d query embedding behind
each box. The boxes are the visual evidence shown to the user; the embeddings
are the structure tokens fed to the language model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union

import cv2
import numpy as np
import torch

from ..constants import (
    DETR_CONF_THRESH,
    DETR_DIM,
    DETR_IMGSZ,
    RT_DETR_CLASSES,
)


@dataclass
class FrameDetections:
    """Detections for a single frame."""

    boxes_xyxy: np.ndarray  # (n, 4) float32, pixel coords in the source frame
    classes: np.ndarray  # (n,) int32, index into RT_DETR_CLASSES
    confidences: np.ndarray  # (n,) float32
    embeddings: np.ndarray  # (n, 256) float32, decoder query per box

    @classmethod
    def empty(cls) -> "FrameDetections":
        return cls(
            boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            classes=np.zeros((0,), dtype=np.int32),
            confidences=np.zeros((0,), dtype=np.float32),
            embeddings=np.zeros((0, DETR_DIM), dtype=np.float32),
        )


class CardiacDetector:
    """Wraps the fine-tuned RT-DETR checkpoint.

    Parameters
    ----------
    checkpoint:
        Path to ``rtdetr_cardiac_7cls.pt``.
    device:
        GPU index or torch device string.
    conf:
        Confidence threshold for keeping a detection.
    """

    def __init__(
        self,
        checkpoint: Union[str, Path],
        device: Union[int, str] = 0,
        conf: float = DETR_CONF_THRESH,
        imgsz: int = DETR_IMGSZ,
    ):
        from ultralytics import RTDETR

        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"detector checkpoint not found: {checkpoint}")

        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.model = RTDETR(str(checkpoint))
        self.class_names = list(getattr(self.model, "names", None) or RT_DETR_CLASSES)

        # Hook the decoder layer the model actually evaluates at, plus its
        # score head, so we can recover the query behind each output box.
        self._queries: Dict[str, torch.Tensor] = {}
        decoder = self.model.model.model[-1]
        eval_idx = decoder.decoder.eval_idx
        self._handles = [
            decoder.decoder.layers[eval_idx].register_forward_hook(
                lambda m, i, o: self._queries.__setitem__("embeddings", o.detach().cpu())
            ),
            decoder.dec_score_head[eval_idx].register_forward_hook(
                lambda m, i, o: self._queries.__setitem__("logits", o.detach().cpu())
            ),
        ]

    def close(self) -> None:
        """Remove the forward hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def detect_frame(self, frame_rgb: np.ndarray) -> FrameDetections:
        """Detect structures in one RGB frame."""
        # Ultralytics treats array input as BGR, matching what it would get from
        # cv2.imread if the frame were on disk.
        results = self.model.predict(
            source=cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
            imgsz=self.imgsz,
            device=self.device,
            conf=self.conf,
            verbose=False,
        )
        boxes = results[0].boxes
        if len(boxes) == 0:
            return FrameDetections.empty()

        classes = boxes.cls.cpu().int()
        confidences = boxes.conf.cpu()
        embeddings = self._match_queries(classes, confidences)

        return FrameDetections(
            boxes_xyxy=boxes.xyxy.cpu().numpy().astype(np.float32),
            classes=classes.numpy().astype(np.int32),
            confidences=confidences.numpy().astype(np.float32),
            embeddings=embeddings.numpy().astype(np.float32),
        )

    def _match_queries(
        self, classes: torch.Tensor, confidences: torch.Tensor
    ) -> torch.Tensor:
        """Recover the decoder query behind each output box.

        Ultralytics returns boxes without telling us which of the 300 decoder
        queries produced each one, so we match greedily: for every detection,
        take the not-yet-used query whose score for that class is closest to the
        reported confidence. Each query is claimed at most once.
        """
        queries = self._queries["embeddings"][0]  # (300, 256)
        probs = self._queries["logits"][0].sigmoid()  # (300, n_classes)

        available = torch.ones(len(queries), dtype=torch.bool)
        matched = []
        for cls_id, conf in zip(classes.tolist(), confidences.tolist()):
            diff = (probs[:, cls_id] - conf).abs()
            diff = diff.masked_fill(~available, float("inf"))
            best = int(diff.argmin())
            available[best] = False
            matched.append(queries[best])
        return torch.stack(matched)

    def detect_clip(self, frames_rgb: Sequence[np.ndarray]) -> List[FrameDetections]:
        """Detect structures in every frame of a clip."""
        return [self.detect_frame(frame) for frame in frames_rgb]


def structure_tokens(detections: Sequence[FrameDetections]) -> np.ndarray:
    """Collapse a clip's per-frame detections into ``(<=7, 256)`` structure tokens.

    One token per structure the detector saw anywhere in the clip: its query
    embeddings are averaged over every frame and every box of that class. Rows
    are ordered by class id, so a row's position identifies the structure.
    Returns ``(0, 256)`` when nothing was detected.
    """
    per_class: Dict[int, List[np.ndarray]] = {}
    for frame in detections:
        for cls_id, embedding in zip(frame.classes, frame.embeddings):
            per_class.setdefault(int(cls_id), []).append(embedding)

    if not per_class:
        return np.zeros((0, DETR_DIM), dtype=np.float32)
    return np.stack(
        [np.mean(per_class[c], axis=0) for c in sorted(per_class)]
    ).astype(np.float32)
