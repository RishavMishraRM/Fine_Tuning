# ==============================================================================
# 17. Odds Ratio Preference Optimization (ORPO) — Standalone Fine-Tuning Script
# ==============================================================================
# Requirements:
#   pip install torch transformers peft trl datasets accelerate
#
# Usage:
#   # Dry-run inspection (default):
#   python fine_tuning_techniques/17_orpo.py
#
#   # Live training on GPU:
#   python fine_tuning_techniques/17_orpo.py --train --model Qwen/Qwen2.5-0.5B
# ==============================================================================
"""
17. ORPO (Odds Ratio Preference Optimization)
==============================================
Introduced by Hong et al. (2024).
Combines Supervised Fine-Tuning and Preference Alignment into a SINGLE training step
without requiring an auxiliary reference model:
    L_{ORPO} = L_{SFT} + lambda * L_{OR}
"""

import argparse
from typing import Dict, List


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "peft", "trl", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def build_orpo_dataset():
    from datasets import Dataset

    samples = [
        {
            "prompt": "User: How do I report an unrecognized payment on my bank statement?",
            "chosen": "Contact your bank's fraud line immediately or lock your card through the official banking app. Note the transaction ID.",
            "rejected": "Unrecognized charges always resolve themselves automatically after a week without needing any action.",
        },
        {
            "prompt": "User: Can the assistant guarantee loan approval?",
            "chosen": "I cannot guarantee loan approval. Approval depends on your credit profile and underwriter assessment.",
            "rejected": "Yes, I guarantee your personal loan will be approved immediately at 0% interest.",
        },
    ]
    return Dataset.from_list(samples)


def setup_orpo_trainer(model_name: str = "mistralai/Mistral-7B-v0.1"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType
    from trl import ORPOConfig, ORPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    dataset = build_orpo_dataset()

    orpo_args = ORPOConfig(
        output_dir="./outputs/orpo_model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=8e-6,
        beta=0.1,
        max_prompt_length=256,
        max_length=512,
        num_train_epochs=3,
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        report_to="none",
    )

    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    return trainer


def main():
    parser = argparse.ArgumentParser(description="17. ORPO")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model identifier")
    parser.add_argument("--output-dir", type=str, default="./outputs/orpo_model", help="Directory to save model")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default: dry-run inspection)")
    args = parser.parse_args()

    print("=" * 75)
    print("17. ODDS RATIO PREFERENCE OPTIMIZATION (ORPO) — STANDALONE MODULE")
    print("=" * 75)
    print(f"• Base Model:       {args.model}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Pipeline:         Single-Stage (SFT + Preference Alignment concurrently)")
    print(f"• Reference Model:  None (saves ~50% VRAM compared to DPO)")
    print(f"• VRAM Requirement: 16 - 24 GB for 7B models")

    missing = check_prerequisites()
    if missing:
        print("\n⚠️  MISSING PREREQUISITES:")
        print(f"  This script requires: {', '.join(missing)}")
        print("\n  To install the required packages:")
        print("      pip install torch transformers peft trl datasets accelerate")
        print("=" * 75)
        return

    if not args.train:
        print("\n💡 Mode: DRY-RUN / INSPECTION ONLY")
        print("  All dependencies are verified! To execute live training on GPU:")
        print(f"      python fine_tuning_techniques/17_orpo.py --train --model {args.model}")
        print("=" * 75)
        return

    print("\n🚀 Starting ORPO single-stage execution...")
    trainer = setup_orpo_trainer(model_name=args.model)
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\n✓ ORPO model saved successfully to: {args.output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
