# ==============================================================================
# 10. Classical RLHF (SFT + RM + PPO) — Standalone Fine-Tuning Script
# ==============================================================================
# Requirements:
#   pip install torch transformers trl accelerate
#
# Usage:
#   # Dry-run inspection (default):
#   python fine_tuning_techniques/10_rlhf_ppo.py
#
#   # Live pipeline setup:
#   python fine_tuning_techniques/10_rlhf_ppo.py --train --model Qwen/Qwen2.5-0.5B
# ==============================================================================
"""
10. Reinforcement Learning from Human Feedback (RLHF)
======================================================
RLHF (Ouyang et al., 2022 / InstructGPT) aligns models through a 3-stage pipeline:

Stage 1: Supervised Fine-Tuning (SFT) on curated instruction demonstrations.
Stage 2: Reward Model (RM) Training:
         A model r_psi(x, y) trained on pairwise preference comparisons:
         L_RM = - E_{(x, y_w, y_l)} [ log sigma( r_psi(x, y_w) - r_psi(x, y_l) ) ]
Stage 3: Reinforcement Learning (PPO) against the learned reward model:
         Objective: max_theta E_{(x, y) ~ pi_theta} [ r_psi(x, y) - beta * D_KL(pi_theta(y|x) || pi_ref(y|x)) ]
         The KL penalty prevents the policy from deviating into reward hacking.
"""

import argparse
from dataclasses import dataclass
from typing import List, Dict, Any


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "trl"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def setup_rlhf_ppo_trainer(policy_model_name: str = "mistralai/Mistral-7B-v0.1"):
    import torch
    from transformers import AutoTokenizer
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

    ppo_config = PPOConfig(
        model_name=policy_model_name,
        learning_rate=1.4e-5,
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=2,
        init_kl_coef=0.2,
        target_kl=0.1,
    )

    tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else None
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        policy_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        policy_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )
    return trainer


def main():
    parser = argparse.ArgumentParser(description="10. Classical RLHF (PPO)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Policy model identifier")
    parser.add_argument("--output-dir", type=str, default="./outputs/rlhf_ppo_model", help="Directory to save model")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default: dry-run inspection)")
    args = parser.parse_args()

    print("=" * 75)
    print("10. CLASSICAL RLHF (SFT + RM + PPO) — STANDALONE MODULE")
    print("=" * 75)
    print(f"• Policy Model:     {args.model}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Architecture:     4-Model System (Actor, Critic, Reference Model, Reward Model)")
    print(f"• KL Regularizer:   Dynamic adaptive KL penalty (beta=0.2)")
    print(f"• VRAM Requirement: 48 - 80 GB VRAM cluster")

    missing = check_prerequisites()
    if missing:
        print("\n⚠️  MISSING PREREQUISITES:")
        print(f"  This script requires: {', '.join(missing)}")
        print("\n  To install the required packages:")
        print("      pip install torch transformers trl accelerate")
        print("=" * 75)
        return

    if not args.train:
        print("\n💡 Mode: DRY-RUN / INSPECTION ONLY")
        print("  All dependencies are verified! To execute live training on GPU:")
        print(f"      python fine_tuning_techniques/10_rlhf_ppo.py --train --model {args.model}")
        print("=" * 75)
        return

    print("\n🚀 Initializing RLHF PPO training pipeline...")
    trainer = setup_rlhf_ppo_trainer(policy_model_name=args.model)
    print("✓ PPO Actor policy and Value Head (Critic) initialized successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
