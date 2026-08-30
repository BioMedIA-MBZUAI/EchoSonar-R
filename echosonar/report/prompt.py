import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

from ..constants import DETR_DIM, REPORT_PLACEHOLDER_TOKEN_ID, TOKEN_DIM
from ..encoders.detector import FrameDetections, structure_tokens

SYSTEM_PROMPT = "You are a medical imaging assistant specialized in echocardiography analysis."

# --- Full report -----------------------------------------------------------
FULL_REPORT_QUESTIONS = (
    "Generate a complete echocardiographic report for this study.",
    "Provide a full structured echocardiographic report including all findings and conclusions.",
    "Write a comprehensive echocardiographic report covering all cardiac structures and conclusions.",
    "Produce a detailed echocardiographic report with findings for each structure and a final conclusion.",
)
FULL_REPORT_QUESTION = FULL_REPORT_QUESTIONS[-1]

# --- Abnormality list (whole-study) -----------------------------------------
ABNORMALITY_LIST_QUESTIONS = (
    "What abnormalities are present in this echocardiographic study?",
    "List all abnormal findings identified in this echocardiogram.",
    "What pathological findings can be identified in this study?",
    "Summarize the abnormalities observed in this echocardiographic examination.",
)
ABNORMALITY_LIST_QUESTION = ABNORMALITY_LIST_QUESTIONS[0]

# --- Conclusion --------------------------------------------------------------
CONCLUSION_QUESTIONS = (
    "What is the overall conclusion of this echocardiographic study?",
    "Provide the conclusion for this echocardiogram.",
    "Summarize the key conclusions from this echocardiographic examination.",
    "What are the main conclusions based on this echocardiographic assessment?",
)
CONCLUSION_QUESTION = CONCLUSION_QUESTIONS[0]

# --- Structure description (per structure) -----------------------------------
STRUCTURE_NAMES: Dict[str, str] = {
    "left_ventricle": "Left Ventricle",
    "right_ventricle": "Right Ventricle",
    "left_atrium": "Left Atrium",
    "right_atrium": "Right Atrium",
    "aortic_valve": "Aortic Valve",
    "mitral_valve": "Mitral Valve",
    "tricuspid_valve": "Tricuspid Valve",
    "pulmonic_valve": "Pulmonic Valve",
    "pericardium": "Pericardium",
    "aortic_root": "Aortic Root",
    "aortic_arch": "Aortic Arch",
    "pulmonary_artery": "Pulmonary Artery",
}


def structure_description_questions(structure: str) -> Tuple[str, str, str, str]:
    """The 4 trained paraphrases asking about one structure's findings.

    ``structure`` is a key of :data:`STRUCTURE_NAMES`, e.g. ``"left_atrium"``.
    """
    name = STRUCTURE_NAMES[structure].lower()
    return (
        f"Describe the {name} findings in this echocardiographic study.",
        f"What are the echocardiographic findings for the {name}?",
        f"Provide a description of the {name} based on this echocardiogram.",
        f"What does the echocardiogram reveal about the {name}?",
    )


# --- Abnormality classification (per abnormality, yes/no) --------------------
ABNORMALITY_QUESTIONS: Dict[str, Tuple[str, str, str, str]] = {
    "Tricuspid Regurgitation": (
        "Is there tricuspid regurgitation present in this echocardiogram?",
        "Does this study show any evidence of tricuspid regurgitation?",
        "Can you identify tricuspid regurgitation in this echocardiographic study?",
        "Is tricuspid valve regurgitation observed?",
    ),
    "Mitral Valve Regurgitation": (
        "Is there mitral valve regurgitation present in this echocardiogram?",
        "Does this study show evidence of mitral regurgitation?",
        "Can you identify mitral valve regurgitation in this echocardiographic study?",
        "Is mitral regurgitation observed?",
    ),
    "Aortic Stenosis": (
        "Is there aortic stenosis present in this echocardiogram?",
        "Does this study show any evidence of aortic stenosis?",
        "Can you identify aortic stenosis in this echocardiographic study?",
        "Is aortic valve stenosis observed?",
    ),
    "Mitral Valve Calcification": (
        "Is there mitral valve calcification present in this echocardiogram?",
        "Does this study show evidence of mitral valve calcification?",
        "Can you identify mitral valve calcification in this echocardiographic study?",
        "Is calcification of the mitral valve observed?",
    ),
    "Right Atrial Enlargement": (
        "Is there right atrial enlargement present in this echocardiogram?",
        "Does this study show evidence of right atrial dilation?",
        "Can you identify right atrial enlargement in this echocardiographic study?",
        "Is the right atrium enlarged?",
    ),
    "Right Ventricular Enlargement": (
        "Is there right ventricular enlargement present in this echocardiogram?",
        "Does this study show evidence of right ventricular dilation?",
        "Can you identify right ventricular enlargement in this echocardiographic study?",
        "Is the right ventricle enlarged?",
    ),
    "Left Atrial Enlargement": (
        "Is there left atrial enlargement present in this echocardiogram?",
        "Does this study show evidence of left atrial dilation?",
        "Can you identify left atrial enlargement in this echocardiographic study?",
        "Is the left atrium enlarged?",
    ),
    "Left Ventricular Systolic Function": (
        "Is there left ventricular systolic dysfunction present in this echocardiogram?",
        "Does this study show evidence of impaired left ventricular systolic function?",
        "Can you identify left ventricular systolic dysfunction in this echocardiographic study?",
        "Is left ventricular systolic function abnormal?",
    ),
    "Left Ventricular Enlargement": (
        "Is there left ventricular enlargement present in this echocardiogram?",
        "Does this study show evidence of left ventricular dilation?",
        "Can you identify left ventricular enlargement in this echocardiographic study?",
        "Is the left ventricle enlarged?",
    ),
    "Bicuspid Aortic Valve": (
        "Is there a bicuspid aortic valve present in this echocardiogram?",
        "Does this study show evidence of bicuspid aortic valve morphology?",
        "Can you identify a bicuspid aortic valve in this echocardiographic study?",
        "Is bicuspid aortic valve morphology observed?",
    ),
    "Aortic Regurgitation": (
        "Is there aortic regurgitation present in this echocardiogram?",
        "Does this study show evidence of aortic regurgitation?",
        "Can you identify aortic regurgitation in this echocardiographic study?",
        "Is aortic valve regurgitation observed?",
    ),
}
ABNORMALITIES: Tuple[str, ...] = tuple(ABNORMALITY_QUESTIONS)


def abnormality_classification_questions(abnormality: str) -> Tuple[str, str, str, str]:
    """The 4 trained paraphrases asking whether ``abnormality`` is present.

    ``abnormality`` must be a key of :data:`ABNORMALITY_QUESTIONS` (see
    :data:`ABNORMALITIES` for the full list, e.g. ``"Tricuspid Regurgitation"``).
    """
    return ABNORMALITY_QUESTIONS[abnormality]


@dataclass
class ReportPrompt:
    input_ids: torch.Tensor  # (L,) long
    clip_features: torch.Tensor  # (N_clip, 768) float32
    clip_positions: torch.Tensor  # (N_clip,) long
    detr_features: torch.Tensor  # (N_detr, 256) float32
    detr_positions: torch.Tensor  # (N_detr,) long
    vision_mask: torch.Tensor  # (L,) bool


def _load_frame_detections(detections_h5: h5py.File, clip_id: str) -> List[FrameDetections]:
    """Reconstruct per-frame detections from ``detections.h5`` for one clip."""
    if clip_id not in detections_h5:
        return []
    group = detections_h5[clip_id]
    frames = []
    for key in sorted(
        (k for k in group.keys() if k.startswith("frame_")),
        key=lambda k: int(k.split("_")[1]),
    ):
        frame = group[key]
        frames.append(
            FrameDetections(
                boxes_xyxy=frame["boxes_xyxy"][:],
                classes=frame["classes"][:],
                confidences=frame["confidences"][:],
                embeddings=frame["embeddings"][:],
            )
        )
    return frames


def select_one_clip_per_view(
    clip_ids: Sequence[str], views: Dict[str, str], seed: Optional[int] = 42
) -> List[str]:
    """Group clips by view, keep one per view, shuffle the order.

    Matches the training dataset's ``_select_dicoms``: a study with several
    clips of the same view is subsampled to one, since that is the input shape
    the model was trained on. Deterministic given ``seed`` (pass ``None`` for
    non-deterministic, matching training exactly).
    """
    rng = random.Random(seed)
    groups: Dict[str, List[str]] = {}
    for clip_id in clip_ids:
        groups.setdefault(views.get(clip_id, "unknown"), []).append(clip_id)

    selected = [rng.choice(group) for group in groups.values()]
    rng.shuffle(selected)
    return selected


def build_report_prompt(
    clip_ids: Sequence[str],
    clip_tokens_h5: h5py.File,
    detections_h5: h5py.File,
    tokenizer,
    views: Optional[Dict[str, str]] = None,
    question: str = FULL_REPORT_QUESTION,
    system: str = SYSTEM_PROMPT,
    one_per_view: bool = True,
    seed: Optional[int] = 42,
) -> ReportPrompt:
    if views is None:
        views = {}
        for clip_id in clip_ids:
            if clip_id not in clip_tokens_h5:
                continue
            attrs = clip_tokens_h5[clip_id].attrs
            views[clip_id] = attrs.get("fine_view") or attrs["view"]

    chosen = select_one_clip_per_view(clip_ids, views, seed=seed) if one_per_view else list(clip_ids)

    all_ids: List[int] = []
    clip_features_list: List[np.ndarray] = []
    detr_features_list: List[np.ndarray] = []
    clip_positions: List[int] = []
    detr_positions: List[int] = []

    def append_text(text: str) -> None:
        all_ids.extend(tokenizer.encode(text, add_special_tokens=False))

    def append_vision(n_tokens: int, features: np.ndarray, positions: List[int]) -> None:
        start = len(all_ids)
        all_ids.extend([REPORT_PLACEHOLDER_TOKEN_ID] * n_tokens)
        positions.extend(range(start, start + n_tokens))

    append_text(f"<|im_start|>system\n{system}<|im_end|>\n")
    append_text("<|im_start|>user\n")

    for clip_id in chosen:
        append_text(f"{views.get(clip_id, 'unknown')}:\n")

        if clip_id in clip_tokens_h5:
            tokens = clip_tokens_h5[clip_id]["tokens"][:]  # (393, 768)
            append_vision(tokens.shape[0], tokens, clip_positions)
            clip_features_list.append(tokens)

        detr_tokens = structure_tokens(_load_frame_detections(detections_h5, clip_id))
        if detr_tokens.shape[0] > 0:
            append_text("Structure features:\n")
            append_vision(detr_tokens.shape[0], detr_tokens, detr_positions)
            detr_features_list.append(detr_tokens)

    append_text(f"{question}<|im_end|>\n")
    append_text("<|im_start|>assistant\n<think>\n")

    seq_len = len(all_ids)
    vision_mask = [False] * seq_len
    for p in clip_positions:
        vision_mask[p] = True
    for p in detr_positions:
        vision_mask[p] = True

    clip_features = (
        np.concatenate(clip_features_list, axis=0)
        if clip_features_list
        else np.zeros((0, TOKEN_DIM), dtype=np.float32)
    )
    detr_features = (
        np.concatenate(detr_features_list, axis=0)
        if detr_features_list
        else np.zeros((0, DETR_DIM), dtype=np.float32)
    )

    return ReportPrompt(
        input_ids=torch.tensor(all_ids, dtype=torch.long),
        clip_features=torch.as_tensor(clip_features, dtype=torch.float32),
        clip_positions=torch.tensor(clip_positions, dtype=torch.long),
        detr_features=torch.as_tensor(detr_features, dtype=torch.float32),
        detr_positions=torch.tensor(detr_positions, dtype=torch.long),
        vision_mask=torch.tensor(vision_mask, dtype=torch.bool),
    )
