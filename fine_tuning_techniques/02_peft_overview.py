# ==============================================================================
# 02. PEFT Architecture Overview — Standalone Analysis Script
# ==============================================================================
# Requirements:
#   pip install torch transformers peft
#
# Usage:
#   python fine_tuning_techniques/02_peft_overview.py [--model MODEL]
# ==============================================================================
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
"""

import argparse
from typing import Dict, Any


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "peft"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def compare_peft_methods(model_name: str = "mistralai/Mistral-7B-v0.1") -> Dict[str, Any]:
    """Demonstrates how different PEFT configs attach to a model and inspects their configs."""
    from peft import (
        TaskType,
        LoraConfig,
        PrefixTuningConfig,
        PromptTuningConfig,
        PromptEncoderConfig,
    )

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
    print(f"{'Method':<16} {'PEFT Type':<20} {'Target / Virtual Tokens':<30}")
    print("-" * 70)
    for name, config in configs.items():
        details = (
            f"r={config.r}, alpha={config.lora_alpha}"
            if hasattr(config, "r")
            else f"tokens={config.num_virtual_tokens}"
        )
        print(f"• {name:<14s} {str(config.peft_type):<20s} {details:<30s}")

    return configs


def inspect_trainable_parameters(model) -> Dict[str, float]:
    """Utility to print and return the count of trainable vs total parameters."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable_params / all_param if all_param > 0 else 0
    print(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_param:,} || "
        f"Trainable%: {pct:.4f}%"
    )
    return {"trainable": trainable_params, "total": all_param, "ratio_pct": pct}


def main():
    parser = argparse.ArgumentParser(description="02. PEFT Architecture Overview")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Target model identifier")
    args = parser.parse_args()

    print("=" * 75)
    print("02. PARAMETER-EFFICIENT FINE-TUNING (PEFT) OVERVIEW")
    print("=" * 75)
    print(f"Target Model: {args.model}")

    missing = check_prerequisites()
    if missing:
        print("\n⚠️  MISSING PREREQUISITES:")
        print(f"  This script requires: {', '.join(missing)}")
        print("\n  To install the required packages:")
        print("      pip install torch transformers peft")
        print("=" * 75)
        return

    compare_peft_methods(args.model)

    print("\nTheoretical Parameter & VRAM Footprint Table (7B Model, 14 GB base FP16):")
    print(f"{'Method':<18} {'Trainable Params':<18} {'Adapter Size':<15} {'VRAM Required'}")
    print("-" * 75)
    print(f"{'Full Fine-Tuning':<18} {'7,000,000,000 (100%)':<18} {'14 GB':<15} {'> 112 GB'}")
    print(f"{'LoRA (r=16)':<18} {'~6,800,000 (0.10%)':<18} {'~27 MB':<15} {'~16-24 GB'}")
    print(f"{'QLoRA (4-bit r=16)':<18} {'~6,800,000 (0.10%)':<18} {'~27 MB':<15} {'~6-8 GB'}")
    print(f"{'Prefix Tuning':<18} {'~1,500,000 (0.02%)':<18} {'~6 MB':<15} {'~12-16 GB'}")
    print(f"{'Prompt Tuning':<18} {'~81,920 (0.001%)':<18} {'~320 KB':<15} {'< 10 GB'}")
    print("=" * 75)


if __name__ == "__main__":
    main()
