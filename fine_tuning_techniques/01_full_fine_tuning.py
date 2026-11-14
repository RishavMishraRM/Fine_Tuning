# ==============================================================================
# 01. Full Fine-Tuning (FFT) — Standalone Fine-Tuning Script
# ==============================================================================
# Requirements:
#   pip install torch transformers datasets accelerate
#
# Usage:
#   # Dry-run inspection (default):
#   python fine_tuning_techniques/01_full_fine_tuning.py
#
#   # Live training on GPU:
#   python fine_tuning_techniques/01_full_fine_tuning.py --train --model Qwen/Qwen2.5-0.5B
# ==============================================================================
"""
01. Full Fine-Tuning (FFT)
===========================
Full Fine-Tuning updates all parameters (weights and biases) of the neural
network during backpropagation:
    W_new = W_old - lr * grad(L)

Key Characteristics:
- High compute & memory cost: Requires ~16 bytes of VRAM per parameter (weights in fp16/bf16,
  gradients in fp16/bf16, and optimizer states in fp32 for AdamW: 2 + 2 + 12 = 16 bytes).
  A 7B parameter model requires ~112 GB VRAM for training without offloading.
- Highest ceiling for capability modification and domain absorption.
- High risk of catastrophic forgetting of out-of-domain general knowledge.
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Optional


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


@dataclass
class FFTConfig:
    model_name: str = "mistralai/Mistral-7B-v0.1"
    output_dir: str = "./outputs/fft_model"
    learning_rate: float = 2e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_seq_length: int = 512
    bf16: bool = True
    gradient_checkpointing: bool = True


def prepare_sample_dataset(tokenizer, max_length: int = 512):
    """Generates a tokenized causal LM dataset for general domain adaptation."""
    from datasets import Dataset

    raw_data = [
        {"text": "User: What is the principle of conservation of energy? Assistant: Energy cannot be created or destroyed, only transformed from one form to another."},
        {"text": "User: How does HTTPS protect web traffic? Assistant: HTTPS uses TLS/SSL encryption to secure communications between browser and server."},
        {"text": "User: Explain database normalization. Assistant: Normalization organizes relational database tables to reduce redundancy and improve data integrity."},
        {"text": "User: What should I do if my bank card is lost? Assistant: Freeze the card immediately via your official mobile banking app and notify card services."},
    ]
    dataset = Dataset.from_list(raw_data)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])
    return tokenized


def build_fft_trainer(config: Optional[FFTConfig] = None):
    """Builds a Full Fine-Tuning Trainer instance."""
    config = config or FFTConfig()
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if (config.bf16 and torch.cuda.is_available()) else torch.float32,
        device_map=device_map,
    )

    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[FFT] Trainable parameters: {trainable_params:,} / {total_params:,} (100.0%)")

    tokenized_dataset = prepare_sample_dataset(tokenizer, config.max_seq_length)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        bf16=config.bf16 and torch.cuda.is_available(),
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    return trainer


def main():
    parser = argparse.ArgumentParser(description="01. Full Fine-Tuning (FFT)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model identifier")
    parser.add_argument("--output-dir", type=str, default="./outputs/fft_model", help="Directory to save model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default: dry-run inspection)")
    args = parser.parse_args()

    print("=" * 75)
    print("01. FULL FINE-TUNING (FFT) — STANDALONE MODULE")
    print("=" * 75)
    print(f"• Base Model:       {args.model}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Learning Rate:    {args.lr}")
    print(f"• Epochs:           {args.epochs}")
    print(f"• Batch Size:       {args.batch_size}")
    print(f"• VRAM Requirement: ~16 bytes/param (~112 GB for 7B; multi-GPU cluster required)")

    missing = check_prerequisites()
    if missing:
        print("\n⚠️  MISSING PREREQUISITES:")
        print(f"  This script requires: {', '.join(missing)}")
        print("\n  To install the required packages:")
        print("      pip install torch transformers datasets accelerate")
        print("\n  To install the complete repository training suite:")
        print("      pip install -r requirements-training.txt")
        print("=" * 75)
        return

    if not args.train:
        print("\n💡 Mode: DRY-RUN / INSPECTION ONLY")
        print("  All dependencies are verified! To execute live training on GPU:")
        print(f"      python fine_tuning_techniques/01_full_fine_tuning.py --train --model {args.model}")
        print("=" * 75)
        return

    print("\n🚀 Starting Full Fine-Tuning execution...")
    config = FFTConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_train_epochs=args.epochs,
    )
    trainer = build_fft_trainer(config)
    trainer.train()
    trainer.save_model(config.output_dir)
    print(f"\n✓ Training complete! Full model weights saved to: {config.output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
