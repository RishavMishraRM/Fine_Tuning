# ==============================================================================
# 05. Supervised Fine-Tuning (SFT) — Standalone Fine-Tuning Script
# ==============================================================================
# Requirements:
#   pip install torch transformers peft trl datasets accelerate
#
# Usage:
#   # Dry-run inspection (default):
#   python fine_tuning_techniques/05_sft.py
#
#   # Live training on GPU:
#   python fine_tuning_techniques/05_sft.py --train --model Qwen/Qwen2.5-0.5B
# ==============================================================================
"""
05. Supervised Fine-Tuning (SFT)
=================================
Supervised Fine-Tuning adapts a base (pre-trained) causal language model to follow
user instructions or conduct dialogue using curated pairs of inputs and target outputs:
    Loss = - sum_{t=1}^{T} log P(y_t | x, y_{<t}; theta)

In standard SFT:
- Tokens in the prompt x are often masked (loss calculated only on completion tokens y).
- TRL's SFTTrainer manages prompt formatting, packing sequences to max context length,
  and computing cross-entropy loss over response tokens.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, Any, List


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "peft", "trl", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


@dataclass
class SFTConfig:
    model_name: str = "mistralai/Mistral-7B-v0.1"
    output_dir: str = "./outputs/sft_model"
    max_seq_length: int = 1024
    learning_rate: float = 2e-5
    batch_size: int = 4
    num_train_epochs: int = 2


def build_sample_sft_dataset():
    """Creates a sample multi-turn / instruction-response dataset."""
    from datasets import Dataset

    samples = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful domain assistant."},
                {"role": "user", "content": "Explain how compound interest works."},
                {"role": "assistant", "content": "Compound interest is calculated on both initial principal and accumulated interest over prior periods."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful domain assistant."},
                {"role": "user", "content": "What is an index fund?"},
                {"role": "assistant", "content": "An index fund is an investment vehicle designed to track a market benchmark like the S&P 500 or Nifty 50."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful domain assistant."},
                {"role": "user", "content": "How do I secure an unauthorized charge on my account?"},
                {"role": "assistant", "content": "Immediately lock your card via your authenticated banking application and contact official customer support."},
            ]
        },
    ]
    return Dataset.from_list(samples)


def setup_sft_trainer(config: SFTConfig):
    """Sets up TRL SFTTrainer with chat template formatting."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, TaskType
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    dataset = build_sample_sft_dataset()

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        max_seq_length=config.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )
    return trainer


def main():
    parser = argparse.ArgumentParser(description="05. Supervised Fine-Tuning (SFT)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model identifier")
    parser.add_argument("--output-dir", type=str, default="./outputs/sft_model", help="Directory to save model")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default: dry-run inspection)")
    args = parser.parse_args()

    print("=" * 75)
    print("05. SUPERVISED FINE-TUNING (SFT) — STANDALONE MODULE")
    print("=" * 75)
    print(f"• Base Model:       {args.model}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Learning Rate:    {args.lr}")
    print(f"• Epochs:           {args.epochs}")
    print(f"• Batch Size:       {args.batch_size}")
    print(f"• Trainer:          TRL SFTTrainer with Conversational Packaging")
    print(f"• VRAM Requirement: 16 - 24 GB for 7B models")

    missing = check_prerequisites()
    if missing:
        print("\n⚠️  MISSING PREREQUISITES:")
        print(f"  This script requires: {', '.join(missing)}")
        print("\n  To install the required packages:")
        print("      pip install torch transformers peft trl datasets accelerate")
        print("\n  To install the complete repository training suite:")
        print("      pip install -r requirements-training.txt")
        print("=" * 75)
        return

    if not args.train:
        print("\n💡 Mode: DRY-RUN / INSPECTION ONLY")
        print("  All dependencies are verified! To execute live training on GPU:")
        print(f"      python fine_tuning_techniques/05_sft.py --train --model {args.model}")
        print("=" * 75)
        return

    print("\n🚀 Starting Supervised Fine-Tuning execution...")
    cfg = SFTConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_train_epochs=args.epochs,
    )
    trainer = setup_sft_trainer(cfg)
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print(f"\n✓ SFT model and adapter saved successfully to: {cfg.output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
