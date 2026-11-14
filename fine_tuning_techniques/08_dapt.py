# ==============================================================================
# 08. Domain-Adaptive Pretraining (DAPT) — Standalone Fine-Tuning Script
# ==============================================================================
# Requirements:
#   pip install torch transformers datasets accelerate
#
# Usage:
#   # Dry-run inspection (default):
#   python fine_tuning_techniques/08_dapt.py
#
#   # Live training on GPU:
#   python fine_tuning_techniques/08_dapt.py --train --model Qwen/Qwen2.5-0.5B
# ==============================================================================
"""
08. Domain-Adaptive Pretraining (DAPT)
=======================================
Introduced by Gururangan et al. (ACL 2020).
DAPT performs continued pretraining on specialized corpora (finance, law, biomedicine)
prior to task-specific supervised fine-tuning.

Key Metric to Monitor:
Domain Perplexity (PPL = exp(CrossEntropyLoss)). A steep drop in domain PPL
indicates the foundation model has internalized the target domain vocabulary.
"""

import argparse
import math
from typing import Dict, List


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def load_domain_corpus():
    from datasets import Dataset

    domain_texts = [
        {"text": "Under statutory financial regulation, statutory liquidity ratio must be maintained in approved securities."},
        {"text": "Non-performing assets are categorized into sub-standard, doubtful, and loss assets based on default duration."},
        {"text": "The credit appraisal process computes debt-service coverage ratio and loan-to-value metrics."},
        {"text": "Automated clearing house transactions settle in accordance with national payment clearing cycles."},
    ]
    return Dataset.from_list(domain_texts)


def evaluate_domain_perplexity(trainer) -> float:
    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss", 0.0)
    perplexity = math.exp(eval_loss)
    print(f"[DAPT] Domain Evaluation Loss: {eval_loss:.4f} || Domain Perplexity: {perplexity:.2f}")
    return perplexity


def setup_dapt_pipeline(model_name: str = "mistralai/Mistral-7B-v0.1"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_domain_corpus()

    def tokenize_fn(x):
        return tokenizer(x["text"], truncation=True, max_length=512, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])

    split_dataset = tokenized_dataset.train_test_split(test_size=0.25, seed=42)

    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )

    training_args = TrainingArguments(
        output_dir="./outputs/dapt_domain_model",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=2e-5,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=default_data_collator,
    )
    return trainer


def main():
    parser = argparse.ArgumentParser(description="08. Domain-Adaptive Pretraining (DAPT)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model identifier")
    parser.add_argument("--output-dir", type=str, default="./outputs/dapt_domain_model", help="Directory to save model")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default: dry-run inspection)")
    args = parser.parse_args()

    print("=" * 75)
    print("08. DOMAIN-ADAPTIVE PRETRAINING (DAPT) — STANDALONE MODULE")
    print("=" * 75)
    print(f"• Base Model:       {args.model}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Evaluation Metric: Domain Perplexity (PPL = exp(CrossEntropyLoss))")
    print(f"• VRAM Requirement: 24 - 48 GB for 7B models")

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
        print(f"      python fine_tuning_techniques/08_dapt.py --train --model {args.model}")
        print("=" * 75)
        return

    print("\n🚀 Starting Domain-Adaptive Pretraining execution...")
    trainer = setup_dapt_pipeline(model_name=args.model)
    trainer.train()
    ppl = evaluate_domain_perplexity(trainer)
    trainer.save_model(args.output_dir)
    print(f"\n✓ DAPT model saved successfully to: {args.output_dir} (Final Perplexity: {ppl:.2f})")
    print("=" * 75)


if __name__ == "__main__":
    main()
