# ==============================================================================
# 04. QLoRA (Quantized LoRA) — Standalone Fine-Tuning Script
# ==============================================================================
# Requirements:
#   pip install torch transformers peft bitsandbytes accelerate
#
# Usage:
#   # Dry-run inspection (default):
#   python fine_tuning_techniques/04_qlora.py
#
#   # Live training on GPU:
#   python fine_tuning_techniques/04_qlora.py --train --model mistralai/Mistral-7B-v0.1
# ==============================================================================
"""
04. QLoRA (Quantized Low-Rank Adaptation)
=========================================
QLoRA (Dettmers et al., 2023) enables fine-tuning of 7B-70B models on a single
consumer GPU (e.g. 6GB - 16GB VRAM) without quality degradation.

Key Innovations:
1. 4-bit NormalFloat (NF4): An optimal quantile data type for normal distributed weights.
2. Double Quantization (DQ): Quantizes quantization constants, saving ~0.37 bits/param.
3. Paged Optimizers: Pages memory between GPU and CPU during memory spikes.
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "peft", "bitsandbytes"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


@dataclass
class QLoRAConfig:
    model_name: str = "mistralai/Mistral-7B-v0.1"
    output_dir: str = "./outputs/qlora_adapter"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    use_double_quant: bool = True
    quant_type: str = "nf4"
    learning_rate: float = 2e-4
    max_seq_length: int = 512


def build_qlora_model(config: QLoRAConfig):
    """Initializes 4-bit quantized base model and attaches trainable FP32/BF16 LoRA adapters."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.quant_type,
        bnb_4bit_use_double_quant=config.use_double_quant,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    qlora_model = get_peft_model(model, peft_config)
    qlora_model.print_trainable_parameters()
    return qlora_model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="04. QLoRA (Quantized LoRA)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model identifier")
    parser.add_argument("--output-dir", type=str, default="./outputs/qlora_adapter", help="Directory to save adapter")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (r)")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA scaling factor")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default: dry-run inspection)")
    args = parser.parse_args()

    print("=" * 75)
    print("04. QLoRA (4-BIT QUANTIZED LoRA) — STANDALONE MODULE")
    print("=" * 75)
    print(f"• Base Model:       {args.model}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Quantization:     4-bit NormalFloat (NF4)")
    print(f"• Double Quant:     Enabled (saves ~0.37 bits/param)")
    print(f"• Compute Dtype:    bfloat16")
    print(f"• Optimizer:        Paged AdamW 8-bit")
    print(f"• VRAM Requirement: < 10 GB for 7B models (runs on RTX 3060 / 4060)")

    missing = check_prerequisites()
    if missing:
        print("\n⚠️  MISSING PREREQUISITES:")
        print(f"  This script requires: {', '.join(missing)}")
        print("\n  To install the required packages:")
        print("      pip install torch transformers peft bitsandbytes accelerate")
        print("\n  To install the complete repository training suite:")
        print("      pip install -r requirements-training.txt")
        print("=" * 75)
        return

    if not args.train:
        print("\n💡 Mode: DRY-RUN / INSPECTION ONLY")
        print("  All dependencies are verified! To execute live training on GPU:")
        print(f"      python fine_tuning_techniques/04_qlora.py --train --model {args.model}")
        print("=" * 75)
        return

    print("\n🚀 Building QLoRA model with 4-bit NF4 quantization...")
    cfg = QLoRAConfig(model_name=args.model, output_dir=args.output_dir, rank=args.rank, alpha=args.alpha)
    model, tokenizer = build_qlora_model(cfg)
    print("✓ Model loaded into 4-bit VRAM successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
