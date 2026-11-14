# ==============================================================================
# 18. Kahneman-Tversky Optimization (KTO) — Standalone Fine-Tuning Script
# ==============================================================================
# Requirements:
#   pip install torch transformers peft trl datasets accelerate
#
# Usage:
#   # Dry-run inspection (default):
#   python fine_tuning_techniques/18_kto.py
#
#   # Live training on GPU:
#   python fine_tuning_techniques/18_kto.py --train --model Qwen/Qwen2.5-0.5B
# ==============================================================================
"""
18. KTO (Kahneman-Tversky Optimization)
========================================
Introduced by Ethayarajh et al. (2024).
Aligns LLMs directly on UNPAIRED binary feedback (thumbs up / thumbs down telemetry)
grounded in behavioral Prospect Theory and loss aversion (penalizing bad completions
more severely than rewarding good ones).
"""

import argparse


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "peft", "trl", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def build_kto_unpaired_dataset():
    from datasets import Dataset

    telemetry_samples = [
        {
            "prompt": "User: How can I block my lost debit card?",
            "completion": "Open your mobile banking app, navigate to Cards, and tap Freeze Card immediately.",
            "label": True,
        },
        {
            "prompt": "User: How can I block my lost debit card?",
            "completion": "Lost cards are not risky. Wait a few weeks to see if someone finds it.",
            "label": False,
        },
        {
            "prompt": "User: What is an OTP used for?",
            "completion": "An OTP is a temporary security code to authenticate transactions. Never disclose it.",
            "label": True,
        },
    ]
    return Dataset.from_list(telemetry_samples)


def setup_kto_trainer(model_name: str = "mistralai/Mistral-7B-v0.1"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType
    from trl import KTOConfig, KTOTrainer

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

    dataset = build_kto_unpaired_dataset()

    kto_args = KTOConfig(
        output_dir="./outputs/kto_model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        beta=0.1,
        desirable_weight=1.0,
        undesirable_weight=1.33,
        max_prompt_length=256,
        max_length=512,
        num_train_epochs=3,
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        report_to="none",
    )

    trainer = KTOTrainer(
        model=model,
        ref_model=None,
        args=kto_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    return trainer


def main():
    parser = argparse.ArgumentParser(description="18. KTO")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model identifier")
    parser.add_argument("--output-dir", type=str, default="./outputs/kto_model", help="Directory to save model")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default: dry-run inspection)")
    args = parser.parse_args()

    print("=" * 75)
    print("18. KAHNEMAN-TVERSKY OPTIMIZATION (KTO) — STANDALONE MODULE")
    print("=" * 75)
    print(f"• Base Model:       {args.model}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Data Format:      Unpaired Binary Telemetry (thumbs up / thumbs down)")
    print(f"• Loss Aversion:    Undesirable Weight = 1.33x (Prospect Theory)")
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
        print(f"      python fine_tuning_techniques/18_kto.py --train --model {args.model}")
        print("=" * 75)
        return

    print("\n🚀 Starting KTO execution...")
    trainer = setup_kto_trainer(model_name=args.model)
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\n✓ KTO model saved successfully to: {args.output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
