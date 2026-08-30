"""Fine-grained echo view classifier -- distinct from EchoPrime's own.

EchoPrime's view classifier (used inside :class:`EchoPrimeEncoder`) predicts
one of 11 coarse views. It has nothing to do with this model: a MobileNetV3-
Small fine-tuned from ImageNet weights on the report-generation training
data's *curated* view labels (``FINE_VIEW_NAMES`` -- things like "PLAX
Standard" or "A4C Zoomed LV"). Its whole purpose is to reproduce those labels
on a new study that has no curated labels of its own, so the text handed to
the report model matches what it saw in training.
"""

from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms

from ..constants import FINE_VIEW_NAMES

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_model(num_classes: int = len(FINE_VIEW_NAMES)) -> torch.nn.Module:
    model = torchvision.models.mobilenet_v3_small()
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(576, 256),
        torch.nn.Hardswish(inplace=True),
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(256, num_classes),
    )
    return model


class FineViewClassifier:
    """Loads ``checkpoints/view_classifier.pt`` and predicts a view label.

    Uses plain ImageNet preprocessing (resize-256 / centre-crop-224 / ImageNet
    mean-std) -- this model was fine-tuned from torchvision's ImageNet
    weights and never saw EchoPrime's pixel-space normalisation, so it must
    not be fed frames preprocessed for the EchoPrime encoder.
    """

    def __init__(self, checkpoint: Union[str, Path], device: Union[str, torch.device] = "cuda"):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"view classifier checkpoint not found: {checkpoint}")

        self.device = torch.device(device)
        self.model = _build_model().to(self.device)
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    @torch.inference_mode()
    def predict_frame(self, frame_rgb: np.ndarray) -> str:
        """Predict the view label for one RGB uint8 frame."""
        x = self.transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(self.device)
        logits = self.model(x)
        return FINE_VIEW_NAMES[int(logits.argmax(dim=1))]

    def predict_clip(self, frames_rgb: Sequence[np.ndarray]) -> str:
        """Predict the view label for a clip, from its middle frame.

        Matches the convention used to label the public-dataset evaluations
        (one prediction per clip, from a single representative frame).
        """
        return self.predict_frame(frames_rgb[len(frames_rgb) // 2])


__all__ = ["FineViewClassifier"]
