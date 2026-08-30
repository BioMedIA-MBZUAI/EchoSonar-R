import os
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

logger = logging.getLogger(__name__)


class EchoSonar(nn.Module):
    """Vision-Language Model: frozen Qwen3-8B + trainable MLP projectors.

    Two projectors map pre-computed vision features into the LLM embedding space:
      - clip_projector:  (B_clip, 768)  -> (B_clip, llm_dim)
      - detr_projector:  (B_detr, 256)  -> (B_detr, llm_dim)
    """

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen3-8B",
        clip_dim: int = 768,
        detr_dim: int = 256,
        torch_dtype: torch.dtype = torch.bfloat16,
        lora_r: int = 16,
        lora_alpha: int = 32,
        full_finetune: bool = False,
    ):
        super().__init__()
        self.full_finetune = full_finetune

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            dtype=torch_dtype,
            attn_implementation="sdpa",
        )

        if full_finetune:
            logger.info("Full fine-tune mode: all LLM parameters are trainable")
        else:
            for param in self.llm.parameters():
                param.requires_grad = False
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, lora_config)

        llm_dim = self.llm.config.hidden_size  # 4096 for Qwen3-8B

        self.clip_projector = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

        self.detr_projector = nn.Sequential(
            nn.LayerNorm(detr_dim),
            nn.Linear(detr_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

        self.clip_projector.to(torch_dtype)
        self.detr_projector.to(torch_dtype)

        # Expose config for HF Trainer / DeepSpeed compatibility
        self.config = self.llm.config

    def _get_embed_tokens(self):
        """Navigate through PEFT/DeepSpeed wrappers to find embed_tokens."""
        m = self.llm
        if hasattr(m, 'base_model'):
            m = m.base_model
        if hasattr(m, 'model') and hasattr(m.model, 'model'):
            # Qwen3ForCausalLM -> .model (Qwen3Model) -> .embed_tokens
            return m.model.model.embed_tokens
        elif hasattr(m, 'model') and hasattr(m.model, 'embed_tokens'):
            return m.model.embed_tokens
        elif hasattr(m, 'embed_tokens'):
            return m.embed_tokens
        raise AttributeError(f"Cannot find embed_tokens in {type(m)}")

    def forward(
        self,
        input_ids: torch.LongTensor,             # (B, L)
        attention_mask: torch.LongTensor,         # (B, L)
        labels: torch.LongTensor,                 # (B, L)
        clip_features: List[torch.Tensor],        # list of B tensors, each (N_clip_i, 768)
        detr_features: List[torch.Tensor],        # list of B tensors, each (N_detr_i, 256)
        clip_positions: List[torch.Tensor],       # list of B tensors, each (N_clip_i,)
        detr_positions: List[torch.Tensor],       # list of B tensors, each (N_detr_i,)
        vision_mask: torch.BoolTensor,            # (B, L)
        **kwargs,
    ):
        B, L = input_ids.shape
        device = input_ids.device
        dtype = next(self.clip_projector.parameters()).dtype

        text_embeds = self._get_embed_tokens()(input_ids)  # (B, L, D)
        vision_embeds = torch.zeros_like(text_embeds)  # (B, L, D)

        for b in range(B):
            if clip_features[b] is not None and clip_features[b].numel() > 0:
                cp = self.clip_projector(clip_features[b].to(device=device, dtype=dtype))
                pos = clip_positions[b].to(device)
                vision_embeds[b, pos] = cp

            if detr_features[b] is not None and detr_features[b].numel() > 0:
                dp = self.detr_projector(detr_features[b].to(device=device, dtype=dtype))
                pos = detr_positions[b].to(device)
                vision_embeds[b, pos] = dp

        vision_mask_3d = vision_mask.unsqueeze(-1).to(device)  # (B, L, 1)
        inputs_embeds = torch.where(vision_mask_3d, vision_embeds, text_embeds)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    def save_full(self, save_dir: str):
        """Save model weights + projectors."""
        os.makedirs(save_dir, exist_ok=True)
        if self.full_finetune:
            self.llm.save_pretrained(os.path.join(save_dir, "llm"))
        else:
            self.llm.save_pretrained(os.path.join(save_dir, "lora_adapter"))
        torch.save(self.clip_projector.state_dict(), os.path.join(save_dir, "clip_projector.pt"))
        torch.save(self.detr_projector.state_dict(), os.path.join(save_dir, "detr_projector.pt"))
        logger.info("Model + projectors saved to %s", save_dir)

    # Keep for backwards compat
    save_projectors = save_full

    def load_from_lora_checkpoint(self, save_dir: str):
        """Load projectors + LoRA from a stage-1 checkpoint, then merge LoRA into base weights.

        Used when transitioning from LoRA SFT to full fine-tune.
        Requires the model to be initialized WITHOUT LoRA (full_finetune=True).
        """
        self.clip_projector.load_state_dict(
            torch.load(os.path.join(save_dir, "clip_projector.pt"), map_location="cpu")
        )
        self.detr_projector.load_state_dict(
            torch.load(os.path.join(save_dir, "detr_projector.pt"), map_location="cpu")
        )
        adapter_path = os.path.join(save_dir, "lora_adapter")
        if os.path.isdir(adapter_path):
            from peft import PeftModel
            self.llm = PeftModel.from_pretrained(self.llm, adapter_path)
            self.llm = self.llm.merge_and_unload()
            logger.info("LoRA merged into base weights from %s", adapter_path)
        else:
            logger.warning("No lora_adapter found in %s, using base weights only", save_dir)
        for param in self.llm.parameters():
            param.requires_grad = True
        logger.info("Full fine-tune: all LLM parameters unfrozen")

    def load_projectors(self, save_dir: str):
        """Load projectors (and LoRA if present) from checkpoint."""
        self.clip_projector.load_state_dict(
            torch.load(os.path.join(save_dir, "clip_projector.pt"), map_location="cpu")
        )
        self.detr_projector.load_state_dict(
            torch.load(os.path.join(save_dir, "detr_projector.pt"), map_location="cpu")
        )
        if self.full_finetune:
            llm_path = os.path.join(save_dir, "llm")
            if os.path.isdir(llm_path):
                from transformers import AutoModelForCausalLM
                self.llm = AutoModelForCausalLM.from_pretrained(
                    llm_path, dtype=next(self.llm.parameters()).dtype,
                    attn_implementation="sdpa",
                )
                for param in self.llm.parameters():
                    param.requires_grad = True
                logger.info("Full LLM weights loaded from %s", llm_path)
            return
        adapter_path = os.path.join(save_dir, "lora_adapter")
        if os.path.isdir(adapter_path):
            adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                import json
                with open(adapter_config_path) as f:
                    adapter_cfg = json.load(f)
                ckpt_r = adapter_cfg.get("r")
                current_r = None
                if hasattr(self.llm, 'peft_config'):
                    current_r = list(self.llm.peft_config.values())[0].r
                if ckpt_r and current_r and ckpt_r != current_r:
                    logger.info("LoRA rank mismatch (model=%d, ckpt=%d), reloading adapter via from_pretrained", current_r, ckpt_r)
                    from peft import PeftModel
                    # Fully strip existing LoRA (merge zeros + unload) to get clean base model
                    base_llm = self.llm.merge_and_unload()
                    self.llm = PeftModel.from_pretrained(base_llm, adapter_path)
                    logger.info("LoRA adapter loaded via from_pretrained from %s", adapter_path)
                else:
                    from safetensors.torch import load_file
                    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
                    bin_path = os.path.join(adapter_path, "adapter_model.bin")
                    if os.path.exists(safetensors_path):
                        state_dict = load_file(safetensors_path)
                    elif os.path.exists(bin_path):
                        state_dict = torch.load(bin_path, map_location="cpu")
                    else:
                        logger.warning("No adapter weights found in %s", adapter_path)
                        return
                    set_peft_model_state_dict(self.llm, state_dict)
            else:
                from safetensors.torch import load_file
                safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
                bin_path = os.path.join(adapter_path, "adapter_model.bin")
                if os.path.exists(safetensors_path):
                    state_dict = load_file(safetensors_path)
                elif os.path.exists(bin_path):
                    state_dict = torch.load(bin_path, map_location="cpu")
                else:
                    logger.warning("No adapter weights found in %s", adapter_path)
                    return
                set_peft_model_state_dict(self.llm, state_dict)
            logger.info("LoRA adapter + projectors loaded from %s", save_dir)
        else:
            logger.info("Projector weights loaded from %s (no LoRA found)", save_dir)

    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing on the LLM to save activation memory.

        Even though the LLM is frozen, activations are still stored for
        backprop through inputs_embeds -> projectors.
        """
        self.llm.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
