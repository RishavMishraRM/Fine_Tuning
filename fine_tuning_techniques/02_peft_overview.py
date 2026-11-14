"""
02. Parameter-Efficient Fine-Tuning (PEFT Overview)
===================================================
PEFT freezes the vast majority of the foundational model parameters and trains
only a tiny subset (typically 0.05% - 2% of the parameters).

The core families of PEFT include:
1. Re-parameterization / Low-Rank: LoRA, QLoRA, DoRA, AdaLoRA
2. Additive Tuning / Bottleneck Adapters: Houlsby, Pfeiffer
3. Soft Prompt / Prefix: Prompt Tuning, Prefix Tuning, P-Tuning v1/v2
4. Selective Tuning: BitFit (tuning only bias terms)

Mathematical Formulation:
Instead of optimizing W in R^{d x k}, PEFT keeps W_0 fixed and optimizes a compact
parameter set theta:
    h = W_0 * x + Delta_W(theta) * x

Benefits:
- Drastically reduced VRAM requirements (optimizer states scale only with theta).
- Storage efficiency: A single base 7B model (14 GB) can host dozens of task adapters (10-50 MB each).
- Mitigates catastrophic forgetting since base weights are immutable.
"""

from dataclasses import dataclass
from typing import Dict, Any
import torch
from peft import (
    TaskType,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptEncoderConfig,
    get_peft_model,
)
from transformers import AutoModelForCausalLM


@dataclass
class PEFTComparisonSummary:
    technique: str
    trainable_ratio_pct: float
    adapter_size_mb: float
    description: str


def compare_peft_methods(model_name: str = "mistralai/Mistral-7B-v0.1") -> Dict[str, Any]:
    """
    Demonstrates how different PEFT configs attach to a model and
    calculates trainable parameter counts.
    """
    print(f"Loading base architecture metadata for: {model_name}...")

    # Define various PEFT configuration types
    configs = {
        "LoRA": LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
        ),
        "Prefix_Tuning": PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=30,
        ),
        "Prompt_Tuning": PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
        ),
        "P_Tuning": PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
            encoder_hidden_size=128,
        ),
    }

    print("\nPEFT Configuration Matrix:")
    for name, config in configs.items():
        print(f"  • {name:<15s}: Type={config.peft_type}, Task={config.task_type}")

    return configs


def inspect_trainable_parameters(model) -> Dict[str, float]:
    """Utility to print and return the count of trainable vs total parameters."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    pct = 100 * trainable_params / all_param if all_param > 0 else 0
    print(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_param:,} || "
        f"Trainable%: {pct:.4f}%"
    )
    return {"trainable": trainable_params, "total": all_param, "ratio_pct": pct}


if __name__ == "__main__":
    print("PEFT Taxonomy & Configuration Framework initialized.")
    compare_peft_methods()
