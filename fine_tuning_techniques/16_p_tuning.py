# ==============================================================================
# 16. P-Tuning (v1 / v2) — Standalone Fine-Tuning Script
# ==============================================================================
# Requirements:
#   pip install torch transformers peft datasets accelerate
#
# Usage:
#   # Dry-run inspection (default):
#   python fine_tuning_techniques/16_p_tuning.py
#
#   # Live training on GPU:
#   python fine_tuning_techniques/16_p_tuning.py --train --model Qwen/Qwen2.5-0.5B
# ==============================================================================
"""
16. P-Tuning (v1 / v2)
=======================
Introduced by Liu et al. (ACL 2022).
Uses a prompt encoder (BiLSTM + 2-layer MLP) to model sequential dependencies
between virtual prompt tokens, overcoming optimization sensitivity in simple prompt tuning.
"""

import argparse


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "peft", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def setup_p_tuning_pipeline(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    num_virtual_tokens: int = 20,
    encoder_hidden_size: int = 128,
):
    import torch
    from datasets import Dataset
    from peft import PromptEncoderConfig, PromptEncoderReparameterizationType, TaskType, get_peft_model
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

    peft_config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=num_virtual_tokens,
        encoder_hidden_size=encoder_hidden_size,
        encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM,
        encoder_num_layers=2,
        encoder_dropout=0.1,
    )

    p_tuned_model = get_peft_model(base_model, peft_config)
    p_tuned_model.print_trainable_parameters()

    sample_data = [
        {"text": "Account update: Customer updated phone number. Category: Profile Modification"},
        {"text": "Account update: International ATM limit increased. Category: Limit Modification"},
    ]
    dataset = Dataset.from_list(sample_data)

    def tokenize_fn(x):
        return tokenizer(x["text"], truncation=True, max_length=128, padding="max_length")

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])

    training_args = TrainingArguments(
        output_dir="./outputs/p_tuning_model",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        num_train_epochs=3,
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        report_to="none",
    )

    trainer = Trainer(
        model=p_tuned_model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=default_data_collator,
    )
    return trainer


def main():
    parser = argparse.ArgumentParser(description="16. P-Tuning (v1 / v2)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model identifier")
    parser.add_argument("--output-dir", type=str, default="./outputs/p_tuning_model", help="Directory to save model")
    parser.add_argument("--virtual-tokens", type=int, default=20, help="Number of virtual tokens")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default: dry-run inspection)")
    args = parser.parse_args()

    print("=" * 75)
    print("16. P-TUNING (v1 / v2) — STANDALONE MODULE")
    print("=" * 75)
    print(f"• Base Model:       {args.model}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Prompt Encoder:   2-Layer BiLSTM + MLP Reparameterization")
    print(f"• Virtual Tokens:   {args.virtual_tokens}")
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
        print(f"      python fine_tuning_techniques/16_p_tuning.py --train --model {args.model}")
        print("=" * 75)
        return

    print("\n🚀 Starting P-Tuning execution...")
    trainer = setup_p_tuning_pipeline(model_name=args.model, num_virtual_tokens=args.virtual_tokens)
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\n✓ P-Tuning model saved successfully to: {args.output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
