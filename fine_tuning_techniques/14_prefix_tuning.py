# ==============================================================================
# 14. Prefix Tuning — Standalone Fine-Tuning Script
# ==============================================================================
# Requirements:
#   pip install torch transformers peft datasets accelerate
#
# Usage:
#   # Dry-run inspection (default):
#   python fine_tuning_techniques/14_prefix_tuning.py
#
#   # Live training on GPU:
#   python fine_tuning_techniques/14_prefix_tuning.py --train --model Qwen/Qwen2.5-0.5B
# ==============================================================================
"""
14. Prefix Tuning
==================
Introduced by Li & Liang (ACL 2021).
Prepends learnable, task-specific continuous virtual Key and Value vectors (P_K, P_V)
directly to the Multi-Head Attention layers across EVERY transformer layer.
Parameter count: ~0.1% of base model weights.
"""

import argparse
from typing import Dict, Any


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "peft", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def setup_prefix_tuning_pipeline(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    num_virtual_tokens: int = 30,
):
    import torch
    from datasets import Dataset
    from peft import PrefixTuningConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else None
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )

    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=num_virtual_tokens,
        prefix_projection=True,
    )

    prefix_model = get_peft_model(base_model, peft_config)
    prefix_model.print_trainable_parameters()

    sample_data = [
        {"text": "Task: Summarize. Input: The fixed deposit matures in 12 months. Output: 1-year deposit tenure."},
        {"text": "Task: Categorize. Input: Customer disputes card charge. Output: Dispute / Chargeback."},
    ]
    dataset = Dataset.from_list(sample_data)

    def tokenize_fn(x):
        return tokenizer(x["text"], truncation=True, max_length=256, padding="max_length")

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])

    training_args = TrainingArguments(
        output_dir="./outputs/prefix_tuning_model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-3,
        num_train_epochs=3,
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        report_to="none",
    )

    trainer = Trainer(
        model=prefix_model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=default_data_collator,
    )
    return trainer


def main():
    parser = argparse.ArgumentParser(description="14. Prefix Tuning")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model identifier")
    parser.add_argument("--output-dir", type=str, default="./outputs/prefix_tuning_model", help="Directory to save model")
    parser.add_argument("--virtual-tokens", type=int, default=30, help="Number of virtual prefix tokens")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default: dry-run inspection)")
    args = parser.parse_args()

    print("=" * 75)
    print("14. PREFIX TUNING — STANDALONE MODULE")
    print("=" * 75)
    print(f"• Base Model:       {args.model}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Virtual Tokens:   {args.virtual_tokens} (prepended to Key/Value in every layer)")
    print(f"• Reparameterize:   MLP projection enabled for numerical stability")
    print(f"• VRAM Requirement: < 12 GB")

    missing = check_prerequisites()
    if missing:
        print("\n⚠️  MISSING PREREQUISITES:")
        print(f"  This script requires: {', '.join(missing)}")
        print("\n  To install the required packages:")
        print("      pip install torch transformers peft datasets accelerate")
        print("=" * 75)
        return

    if not args.train:
        print("\n💡 Mode: DRY-RUN / INSPECTION ONLY")
        print("  All dependencies are verified! To execute live training on GPU:")
        print(f"      python fine_tuning_techniques/14_prefix_tuning.py --train --model {args.model}")
        print("=" * 75)
        return

    print("\n🚀 Starting Prefix Tuning execution...")
    trainer = setup_prefix_tuning_pipeline(model_name=args.model, num_virtual_tokens=args.virtual_tokens)
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\n✓ Prefix-tuned adapter saved successfully to: {args.output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
