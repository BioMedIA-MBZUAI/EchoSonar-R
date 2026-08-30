#!/usr/bin/env python
"""
Needs ``clip_tokens.h5`` (scripts/extract_echoprime_tokens.py) and
``detections.h5`` (scripts/extract_detections.py) already extracted for the
study -- this script only builds the prompt and runs generation.

Full report (the default task)::

    python scripts/generate_report.py \
        --study my_study/ \
        --clip-tokens outputs/clip_tokens.h5 \
        --detections outputs/detections.h5 \
        --checkpoint /path/to/sft_checkpoint \
        --config config.yaml

Abnormality list, conclusion, or a per-structure/per-abnormality question::

    python scripts/generate_report.py --study my_study/ --task abnormality_list ...
    python scripts/generate_report.py --study my_study/ --task conclusion ...
    python scripts/generate_report.py --study my_study/ --task structure_description \
        --structure left_atrium ...
    python scripts/generate_report.py --study my_study/ --task abnormality_classification \
        --abnormality "Tricuspid Regurgitation" ...
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import EchoSonar  # noqa: E402
from echosonar.report import (  # noqa: E402
    ABNORMALITIES,
    ABNORMALITY_LIST_QUESTION,
    CONCLUSION_QUESTION,
    FULL_REPORT_QUESTION,
    STRUCTURE_NAMES,
    abnormality_classification_questions,
    build_report_prompt,
    structure_description_questions,
)
from echosonar.study import discover_studies, load_manifest  # noqa: E402

logger = logging.getLogger("echosonar")

# Tasks that need no further parameter -- their question is fixed.
FIXED_TASK_QUESTIONS = {
    "full_report": FULL_REPORT_QUESTION,
    "abnormality_list": ABNORMALITY_LIST_QUESTION,
    "conclusion": CONCLUSION_QUESTION,
}
# structure_description and abnormality_classification are parameterised --
# see --structure / --abnormality below -- so they aren't in this dict.
TASK_MAX_NEW_TOKENS = {
    "full_report": 4096,
    "abnormality_list": 1024,
    "conclusion": 512,
    "structure_description": 256,
    "abnormality_classification": 128,
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate an echo report with a trained EchoSonar checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--study", help="one study directory of echo clips")
    source.add_argument("--manifest", help="JSONL manifest naming exactly one study")

    p.add_argument("--clip-tokens", required=True, help="clip_tokens.h5 covering this study")
    p.add_argument("--detections", required=True, help="detections.h5 covering this study")
    p.add_argument("--checkpoint", required=True, help="trained projector/LoRA checkpoint dir")
    p.add_argument("--config", default="config.yaml")

    p.add_argument(
        "--task",
        default="full_report",
        choices=sorted(TASK_MAX_NEW_TOKENS),
        help="which trained-on question to ask (default: full_report)",
    )
    p.add_argument(
        "--structure",
        choices=sorted(STRUCTURE_NAMES),
        help="required with --task structure_description",
    )
    p.add_argument(
        "--abnormality",
        choices=ABNORMALITIES,
        help="required with --task abnormality_classification; see echosonar.report.ABNORMALITIES",
    )
    p.add_argument(
        "--question",
        default=None,
        help="use this exact question instead of --task's default (only "
        "in-distribution if it's one of the training paraphrases, see "
        "echosonar.report.prompt)",
    )
    p.add_argument("--no-thinking", action="store_true", help="skip the <think> stage")
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="default depends on --task, see TASK_MAX_NEW_TOKENS",
    )
    p.add_argument("--seed", type=int, default=42, help="one-clip-per-view sampling seed")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", help="write the generated report here (default: stdout)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    studies = load_manifest(args.manifest) if args.manifest else discover_studies(args.study)
    if len(studies) != 1:
        raise ValueError(f"expected exactly one study, got {len(studies)}")
    study = studies[0]
    clip_ids = [clip.clip_id for clip in study.clips]
    logger.info("study %s: %d clips", study.study_id, len(clip_ids))

    device = torch.device(args.device)
    model = EchoSonar(
        llm_name=cfg["model_name"],
        clip_dim=cfg["clip_dim"],
        detr_dim=cfg["detr_dim"],
        lora_r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        full_finetune=cfg.get("full_finetune", True),
    )
    model.load_projectors(args.checkpoint)
    model = model.to(device).eval()
    logger.info("loaded checkpoint from %s", args.checkpoint)

    if args.question:
        question = args.question
    elif args.task == "structure_description":
        if not args.structure:
            raise ValueError("--task structure_description requires --structure")
        question = structure_description_questions(args.structure)[0]
    elif args.task == "abnormality_classification":
        if not args.abnormality:
            raise ValueError("--task abnormality_classification requires --abnormality")
        question = abnormality_classification_questions(args.abnormality)[0]
    else:
        question = FIXED_TASK_QUESTIONS[args.task]
    max_new_tokens = args.max_new_tokens or TASK_MAX_NEW_TOKENS[args.task]

    with h5py.File(args.clip_tokens, "r") as clip_h5, h5py.File(args.detections, "r") as detr_h5:
        prompt = build_report_prompt(
            clip_ids,
            clip_h5,
            detr_h5,
            tokenizer,
            question=question,
            seed=args.seed,
        )

    input_ids = prompt.input_ids.unsqueeze(0).to(device)
    vision_mask = prompt.vision_mask.unsqueeze(0).to(device)

    if not args.no_thinking:
        think_ids = tokenizer.encode("<think>\n", add_special_tokens=False)
        input_ids = torch.cat(
            [input_ids, torch.tensor([think_ids], dtype=torch.long, device=device)], dim=1
        )
        vision_mask = torch.cat(
            [vision_mask, torch.zeros(1, len(think_ids), dtype=torch.bool, device=device)], dim=1
        )
    attention_mask = torch.ones_like(input_ids)

    dtype = next(model.clip_projector.parameters()).dtype
    with torch.no_grad():
        text_embeds = model._get_embed_tokens()(input_ids)
        vision_embeds = torch.zeros_like(text_embeds)

        clip_feat = prompt.clip_features.to(device=device, dtype=dtype)
        if clip_feat.numel() > 0:
            vision_embeds[0, prompt.clip_positions.to(device)] = model.clip_projector(clip_feat)

        detr_feat = prompt.detr_features.to(device=device, dtype=dtype)
        if detr_feat.numel() > 0:
            vision_embeds[0, prompt.detr_positions.to(device)] = model.detr_projector(detr_feat)

        inputs_embeds = torch.where(vision_mask.unsqueeze(-1), vision_embeds, text_embeds)

        eos_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
        eos_ids = [t for t in eos_ids if t is not None]

        generated = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.pad_token_id or 151643,
            eos_token_id=eos_ids,
        )

    full_text = tokenizer.decode(generated[0], skip_special_tokens=False)
    if "</think>" in full_text:
        thinking, answer = full_text.split("</think>", 1)
        thinking = thinking.split("<think>")[-1].strip()
        answer = answer.replace("<|im_end|>", "").strip()
    else:
        thinking, answer = full_text, "(no </think> found; raw output above)"

    report = f"--- Thinking ---\n{thinking}\n\n--- Answer ---\n{answer}\n"
    if args.out:
        Path(args.out).write_text(report)
        logger.info("written to %s", args.out)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
