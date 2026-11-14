# ==============================================================================
# 03. LoRA (Low-Rank Adaptation) — Standalone Fine-Tuning Script
# ==============================================================================
# Requirements:
#   pip install torch transformers peft datasets accelerate
#
# Usage:
#   # Dry-run inspection (default):
#   python fine_tuning_techniques/03_lora.py
#
#   # Live training on GPU:
#   python fine_tuning_techniques/03_lora.py --train --model Qwen/Qwen2.5-0.5B --rank 16
# ==============================================================================
"""
03. LoRA (Low-Rank Adaptation)
===============================
LoRA freezes the pre-trained model weights W_0 in R^{d x k} and injects trainable
rank decomposition matrices into the transformer layers:
    W = W_0 + Delta_W = W_0 + (alpha / r) * (B * A)

Where:
- A in R^{r x k} is initialized with a random Gaussian distribution N(0, sigma^2).
- B in R^{d x r} is initialized to 0, ensuring Delta_W = 0 at the start of training.
- r << min(d, k) is the rank (typically 8, 16, 32, or 64).
- alpha is a constant scaling factor. Scaling by (alpha / r) stabilizes learning
  when varying rank r.

Advantages:
- Reduces trainable parameters by ~10,000x compared to full fine-tuning.
- No inference latency: During deployment, W_merged = W_0 + (alpha / r) * (B * A)
  can be permanently folded back into the original weights.
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "peft", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


@dataclass
class LoRAExperimentConfig:
    model_name: str = "mistralai/Mistral-7B-v0.1"
    output_dir: str = "./outputs/lora_adapter"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    learning_rate: float = 2e-4
    batch_size: int = 4
    num_train_epochs: int = 3


def get_default_lora_targets() -> List[str]:
    """Target modules typically adapted in contemporary transformer architectures."""
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_sample_dataset(tokenizer, max_length: int = 512):
    from datasets import Dataset

    samples = [
        {"text": "Instruction: Summarize key benefits of LoRA. Response: LoRA freezes the base model, adds small rank decomposition matrices, and eliminates inference latency after weight merging."},
        {"text": "Instruction: What is temperature in LLM sampling? Response: Temperature scales the logits before softmax; 0 produces deterministic greedy decoding, while higher values increase creativity."},
        {"text": "Instruction: How does nucleus sampling work? Response: Top-p sampling selects the smallest set of tokens whose cumulative probability exceeds threshold p."},
        {"text": "Instruction: Provide guidance for a lost card. Response: Freeze the card immediately via the banking application to protect against fraudulent charges."},
    ]
    dataset = Dataset.from_list(samples)

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


def build_lora_model(config: LoRAExperimentConfig):
    """Loads base model and attaches LoRA low-rank adapter matrices."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    targets = config.target_modules or get_default_lora_targets()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else None
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=targets,
        bias="none",
    )

    lora_model = get_peft_model(base_model, peft_config)
    lora_model.print_trainable_parameters()
    return lora_model, tokenizer


def merge_and_export_lora(lora_model, save_path: str):
    """Folds low-rank delta weights back into base model for zero-overhead inference."""
    print(f"Merging LoRA weights into base model and exporting to {save_path}...")
    merged_model = lora_model.merge_and_unload()
    merged_model.save_pretrained(save_path)
    print("Merged model exported successfully.")


def main():
    parser = argparse.ArgumentParser(description="03. LoRA (Low-Rank Adaptation)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model identifier")
    parser.add_argument("--output-dir", type=str, default="./outputs/lora_adapter", help="Directory to save adapter")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (r)")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA scaling factor (alpha)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default: dry-run inspection)")
    args = parser.parse_args()

    print("=" * 75)
    print("03. LoRA (LOW-RANK ADAPTATION) — STANDALONE MODULE")
    print("=" * 75)
    print(f"• Base Model:       {args.model}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Rank (r):         {args.rank}")
    print(f"• Alpha:            {args.alpha} (Scaling: {args.alpha / args.rank:.2f}x)")
    print(f"• Target Modules:   {get_default_lora_targets()}")
    print(f"• Trainable Ratio:  ~0.1% of base weights")
    print(f"• VRAM Requirement: 16 - 24 GB for 7B (or 4-6 GB for 0.5B model)")

    missing = check_prerequisites()
    if missing:
        print("\n⚠️  MISSING PREREQUISITES:")
        print(f"  This script requires: {', '.join(missing)}")
        print("\n  To install the required packages:")
        print("      pip install torch transformers peft datasets accelerate")
        print("\n  To install the complete repository training suite:")
        print("      pip install -r requirements-training.txt")
        print("=" * 75)
        return

    if not args.train:
        print("\n💡 Mode: DRY-RUN / INSPECTION ONLY")
        print("  All dependencies are verified! To execute live training on GPU:")
        print(f"      python fine_tuning_techniques/03_lora.py --train --model {args.model}")
        print("=" * 75)
        return

    print("\n🚀 Initializing LoRA training pipeline...")
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    import torch

    cfg = LoRAExperimentConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_train_epochs=args.epochs,
    )

    lora_model, tokenizer = build_lora_model(cfg)
    train_dataset = build_sample_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    lora_model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"\n✓ LoRA adapter weights saved successfully to: {cfg.output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
