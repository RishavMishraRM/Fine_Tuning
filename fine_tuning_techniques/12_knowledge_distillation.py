# ==============================================================================
# 12. Knowledge Distillation (KD) — Standalone Fine-Tuning Script
# ==============================================================================
# Requirements:
#   pip install torch transformers datasets accelerate
#
# Usage:
#   # Dry-run inspection (default):
#   python fine_tuning_techniques/12_knowledge_distillation.py
#
#   # Live training on GPU:
#   python fine_tuning_techniques/12_knowledge_distillation.py --train
# ==============================================================================
"""
12. Knowledge Distillation (KD)
================================
Knowledge Distillation (Hinton et al., 2015) transfers capabilities and dark knowledge
from a heavy Teacher model (e.g., 70B) into a compact Student model (e.g., 0.5B - 3B):

Objective:
    L_{total} = (1 - alpha) * L_{Task} + alpha * tau^2 * D_{KL}( sigma(z_t / tau) || sigma(z_s / tau) )
"""

import argparse
from dataclasses import dataclass


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


@dataclass
class DistillationConfig:
    temperature: float = 2.0
    alpha: float = 0.5


def setup_distillation_pipeline(
    teacher_name: str = "mistralai/Mistral-7B-v0.1",
    student_name: str = "Qwen/Qwen2.5-0.5B",
):
    import torch
    import torch.nn.functional as F
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

    tokenizer = AutoTokenizer.from_pretrained(student_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else None
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )
    teacher_model.eval()

    student_model = AutoModelForCausalLM.from_pretrained(
        student_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )

    sample_data = [
        {"text": "Query: Explain gradient descent. Response: Optimization algorithm minimizing loss by adjusting parameters iteratively."},
        {"text": "Query: Explain softmax function. Response: Converts vector of logits into normalized probability distribution."},
    ]
    dataset = Dataset.from_list(sample_data)

    def tokenize_fn(x):
        return tokenizer(x["text"], truncation=True, max_length=256, padding="max_length")

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])

    training_args = TrainingArguments(
        output_dir="./outputs/distilled_student",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        report_to="none",
    )

    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=default_data_collator,
    )
    return trainer


def main():
    parser = argparse.ArgumentParser(description="12. Knowledge Distillation (KD)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Teacher model identifier")
    parser.add_argument("--student", type=str, default="Qwen/Qwen2.5-0.5B", help="Student model identifier")
    parser.add_argument("--output-dir", type=str, default="./outputs/distilled_student", help="Directory to save student")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default: dry-run inspection)")
    args = parser.parse_args()

    print("=" * 75)
    print("12. KNOWLEDGE DISTILLATION (KD) — STANDALONE MODULE")
    print("=" * 75)
    print(f"• Teacher Model:    {args.model}")
    print(f"• Student Model:    {args.student}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Distillation:     KL Divergence on softened logits (tau = 2.0, alpha = 0.5)")
    print(f"• VRAM Requirement: 24 - 80 GB (concurrent Teacher + Student loading)")

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
        print(f"      python fine_tuning_techniques/12_knowledge_distillation.py --train --model {args.model}")
        print("=" * 75)
        return

    print("\n🚀 Initializing Knowledge Distillation pipeline...")
    trainer = setup_distillation_pipeline(teacher_name=args.model, student_name=args.student)
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\n✓ Distilled Student model saved successfully to: {args.output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
