# ==============================================================================
# 07. Continued Pretraining (CPT) — Standalone Fine-Tuning Script
# ==============================================================================
# Requirements:
#   pip install torch transformers datasets accelerate
#
# Usage:
#   # Dry-run inspection (default):
#   python fine_tuning_techniques/07_cpt.py
#
#   # Live training on GPU:
#   python fine_tuning_techniques/07_cpt.py --train --model Qwen/Qwen2.5-0.5B
# ==============================================================================
"""
07. Continued Pretraining (CPT)
================================
Continued Pretraining trains a completed foundation model on raw, unstructured
text using the original unsupervised next-token prediction objective:
    L_{CPT} = - sum_{i=1}^{N} log P(w_i | w_1, ..., w_{i-1}; theta)

Key Characteristics:
- Ingests raw, unformatted text without instruction/response delimiters.
- Chunks continuous token streams into uniform blocks (block_size=512/2048)
  for 100% token packing efficiency without padding waste.
- Uses conservative learning rates (1e-5 to 3e-5) with cosine decay.
"""

import argparse
from itertools import chain
from typing import Dict, List


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def build_raw_corpus():
    """Simulates raw enterprise unstructured documents."""
    from datasets import Dataset

    documents = [
        {"text": "The central banking authority governs monetary policy and regulates commercial credit mechanisms."},
        {"text": "Liquidity facilities include repo operations, statutory reserves, and marginal lending windows."},
        {"text": "Capital adequacy frameworks ensure that financial institutions maintain sufficient loss-absorbing capital."},
        {"text": "Decentralized consensus algorithms achieve distributed state consistency across Byzantine networks."},
    ]
    return Dataset.from_list(documents)


def group_texts(examples: Dict[str, List[int]], block_size: int = 512):
    """Concatenates continuous tokens and splits into uniform block_size sequences."""
    concatenated = list(chain(*examples["input_ids"]))
    total_length = len(concatenated)

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    chunks = [
        concatenated[i : i + block_size]
        for i in range(0, total_length, block_size)
    ]

    return {
        "input_ids": chunks,
        "labels": [chunk.copy() for chunk in chunks],
    }


def setup_cpt_trainer(model_name: str = "mistralai/Mistral-7B-v0.1", block_size: int = 512):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = build_raw_corpus()

    def tokenize_fn(example):
        return tokenizer(example["text"])

    tokenized = raw_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    lm_dataset = tokenized.map(lambda x: group_texts(x, block_size=block_size), batched=True)

    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )

    training_args = TrainingArguments(
        output_dir="./outputs/cpt_model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=default_data_collator,
    )
    return trainer


def main():
    parser = argparse.ArgumentParser(description="07. Continued Pretraining (CPT)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model identifier")
    parser.add_argument("--output-dir", type=str, default="./outputs/cpt_model", help="Directory to save model")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default: dry-run inspection)")
    args = parser.parse_args()

    print("=" * 75)
    print("07. CONTINUED PRETRAINING (CPT) — STANDALONE MODULE")
    print("=" * 75)
    print(f"• Base Model:       {args.model}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Sequence Packing: Continuous block concatenation (block_size=512)")
    print(f"• Objective:        Causal Next-Token Prediction (Self-Supervised)")
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
        print(f"      python fine_tuning_techniques/07_cpt.py --train --model {args.model}")
        print("=" * 75)
        return

    print("\n🚀 Starting Continued Pretraining execution...")
    trainer = setup_cpt_trainer(model_name=args.model)
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\n✓ Continual pretrained model saved successfully to: {args.output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
