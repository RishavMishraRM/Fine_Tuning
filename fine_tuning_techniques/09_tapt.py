# ==============================================================================
# 09. Task-Adaptive Pretraining (TAPT) — Standalone Fine-Tuning Script
# ==============================================================================
# Requirements:
#   pip install torch transformers datasets accelerate
#
# Usage:
#   # Dry-run inspection (default):
#   python fine_tuning_techniques/09_tapt.py
#
#   # Live training on GPU:
#   python fine_tuning_techniques/09_tapt.py --train --model Qwen/Qwen2.5-0.5B
# ==============================================================================
"""
09. Task-Adaptive Pretraining (TAPT)
=====================================
Gururangan et al. (ACL 2020) demonstrated that pretraining directly on a small,
hyper-targeted unlabeled task corpus (e.g. 5,000 - 50,000 raw customer inquiries)
often yields higher downstream task accuracy gains than pretraining on broad domain data.
"""

import argparse
from typing import List


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def load_unlabeled_task_corpus():
    from datasets import Dataset

    task_queries = [
        {"text": "Customer query: Card transaction failed but amount was deducted from balance."},
        {"text": "Customer query: How can I change the registered mobile phone number for OTP authentication?"},
        {"text": "Customer query: Foreclosure letter generation error on active loan account."},
        {"text": "Customer query: Merchant initiated refund reference ARN has not reflected in statement."},
    ]
    return Dataset.from_list(task_queries)


def setup_tapt_pipeline(model_name: str = "mistralai/Mistral-7B-v0.1"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    task_corpus = load_unlabeled_task_corpus()

    def tokenize_fn(example):
        return tokenizer(example["text"], truncation=True, max_length=256, padding="max_length")

    tokenized_dataset = task_corpus.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])

    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )

    training_args = TrainingArguments(
        output_dir="./outputs/tapt_task_model",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        num_train_epochs=3,
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=default_data_collator,
    )
    return trainer


def main():
    parser = argparse.ArgumentParser(description="09. Task-Adaptive Pretraining (TAPT)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model identifier")
    parser.add_argument("--output-dir", type=str, default="./outputs/tapt_task_model", help="Directory to save model")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default: dry-run inspection)")
    args = parser.parse_args()

    print("=" * 75)
    print("09. TASK-ADAPTIVE PRETRAINING (TAPT) — STANDALONE MODULE")
    print("=" * 75)
    print(f"• Base Model:       {args.model}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Distribution:     Hyper-targeted unlabeled task inquiries")
    print(f"• Learning Rate:    1e-5 (ultra-conservative tuning)")
    print(f"• VRAM Requirement: 16 - 24 GB for 7B models")

    missing = check_prerequisites()
    if missing:
        print("\n⚠️  MISSING PREREQUISITES:")
        print(f"  This script requires: {', '.join(missing)}")
        print("\n  To install the required packages:")
        print("      pip install torch transformers datasets accelerate")
        print("=" * 75)
        return

    if not args.train:
        print("\n💡 Mode: DRY-RUN / INSPECTION ONLY")
        print("  All dependencies are verified! To execute live training on GPU:")
        print(f"      python fine_tuning_techniques/09_tapt.py --train --model {args.model}")
        print("=" * 75)
        return

    print("\n🚀 Starting Task-Adaptive Pretraining execution...")
    trainer = setup_tapt_pipeline(model_name=args.model)
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\n✓ TAPT model saved successfully to: {args.output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
