# ==============================================================================
# 11. Direct Preference Optimization (DPO) — Standalone Fine-Tuning Script
# ==============================================================================
# Requirements:
#   pip install torch transformers peft trl datasets accelerate
#
# Usage:
#   # Dry-run inspection (default):
#   python fine_tuning_techniques/11_dpo.py
#
#   # Live training on GPU:
#   python fine_tuning_techniques/11_dpo.py --train --model Qwen/Qwen2.5-0.5B
# ==============================================================================
"""
11. Direct Preference Optimization (DPO)
=========================================
Introduced by Rafailov et al. (NeurIPS 2023).

Breakthrough Insight:
RLHF requires training a separate reward model r(x, y) and performing unstable,
memory-intensive PPO reinforcement learning with 4 concurrent models.

Rafailov et al. mathematically proved that the optimal policy under the KL-constrained
RL objective has a closed-form solution:
    r*(x, y) = beta * log( pi_theta(y|x) / pi_ref(y|x) )

Substituting this directly into the Bradley-Terry preference likelihood yields the DPO loss:
    L_DPO = - E_{(x, y_w, y_l)} [ log sigma( beta * log( pi_theta(y_w|x)/pi_ref(y_w|x) )
                                          - beta * log( pi_theta(y_l|x)/pi_ref(y_l|x) ) ) ]

Benefits:
- Eliminates the Reward Model completely.
- Eliminates sampling during training (offline supervised loss).
- Far more stable and memory-efficient than PPO.
"""

import argparse
from typing import Dict, List


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "peft", "trl", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def build_preference_dataset():
    from datasets import Dataset

    data = [
        {
            "prompt": "User: What should I do if my payment card is stolen?",
            "chosen": "Immediately lock your card via your official mobile banking application and contact the card issuer hotline. Do not disclose your PIN or OTP to anyone.",
            "rejected": "Share your card number and PIN with me so I can cancel the card for you directly.",
        },
        {
            "prompt": "User: When will my merchant refund reflect in my account?",
            "chosen": "Refund processing times depend on merchant settlement and banking clearing cycles. Please verify the refund ARN reference with your merchant.",
            "rejected": "Your refund is guaranteed to arrive in your bank account tomorrow morning without exception.",
        },
        {
            "prompt": "User: Can you explain how HTTPS encryption operates?",
            "chosen": "HTTPS uses asymmetric TLS cryptography to establish a secure handshake, followed by symmetric AES encryption for session data transmission.",
            "rejected": "HTTPS just sends normal text across the internet through a private hidden tunnel.",
        },
    ]
    return Dataset.from_list(data)


def setup_dpo_trainer(model_name: str = "mistralai/Mistral-7B-v0.1"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, TaskType
    from trl import DPOTrainer

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

    dataset = build_preference_dataset()

    training_args = TrainingArguments(
        output_dir="./outputs/dpo_model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_train_epochs=3,
        warmup_ratio=0.1,
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=0.1,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_length=512,
        max_prompt_length=256,
    )
    return dpo_trainer


def main():
    parser = argparse.ArgumentParser(description="11. Direct Preference Optimization (DPO)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model identifier")
    parser.add_argument("--output-dir", type=str, default="./outputs/dpo_model", help="Directory to save model")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default: dry-run inspection)")
    args = parser.parse_args()

    print("=" * 75)
    print("11. DIRECT PREFERENCE OPTIMIZATION (DPO) — STANDALONE MODULE")
    print("=" * 75)
    print(f"• Base Model:       {args.model}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Alignment Style:  Offline Pairwise (prompt, chosen, rejected)")
    print(f"• Implicit Margin:  beta = 0.1")
    print(f"• VRAM Requirement: 16 - 24 GB for 7B models (with LoRA)")

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
        print(f"      python fine_tuning_techniques/11_dpo.py --train --model {args.model}")
        print("=" * 75)
        return

    print("\n🚀 Starting Direct Preference Optimization execution...")
    trainer = setup_dpo_trainer(model_name=args.model)
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\n✓ DPO aligned model saved successfully to: {args.output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
