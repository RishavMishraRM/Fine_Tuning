# ==============================================================================
# 20. Group Relative Policy Optimization (GRPO) — Standalone Fine-Tuning Script
# ==============================================================================
# Requirements:
#   pip install torch transformers peft trl datasets accelerate
#
# Usage:
#   # Dry-run inspection (default):
#   python fine_tuning_techniques/20_grpo.py
#
#   # Live training on GPU:
#   python fine_tuning_techniques/20_grpo.py --train --model Qwen/Qwen2.5-0.5B
# ==============================================================================
"""
20. GRPO (Group Relative Policy Optimization)
==============================================
Pioneered by DeepSeek (Shao et al., 2024; DeepSeek-R1, 2025).

The Fundamental Bottleneck of PPO:
Standard PPO requires maintaining a separate Critic (Value Network V_phi) of comparable
parameter scale to the Actor policy.

The GRPO Innovation:
GRPO completely ELIMINATES the Critic model.
Instead of learning a baseline V(s), GRPO generates a GROUP of G candidate completions
{o_1, o_2, ..., o_G} for each input prompt q.
It scores each candidate via a reward function r_i = R(q, o_i), and calculates the
Advantage RELATIVE to the group statistics:
    A_i = ( r_i - mean({r_1, ..., r_G}) ) / ( std({r_1, ..., r_G}) + eps )

Impact:
- Reduces RL training memory overhead by ~50%.
- Powered the autonomous reasoning emergence in DeepSeek-R1 and DeepSeekMath.
"""

import argparse
from typing import List, Dict, Callable
import torch


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "peft", "trl", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def compute_group_relative_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Computes group-relative normalized advantages without a Critic/Value network."""
    mean_r = rewards.mean(dim=-1, keepdim=True)
    std_r = rewards.std(dim=-1, keepdim=True)
    return (rewards - mean_r) / (std_r + eps)


def setup_grpo_framework(model_name: str = "mistralai/Mistral-7B-v0.1", group_size: int = 4):
    from transformers import AutoTokenizer
    from datasets import Dataset
    from peft import LoraConfig, TaskType

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        from trl import GRPOConfig, GRPOTrainer

        def correctness_reward_func(prompts, completions, answer, **kwargs):
            return [1.0 if a.strip().lower() in c.strip().lower() else 0.0 for c, a in zip(completions, answer)]

        training_args = GRPOConfig(
            output_dir="./outputs/grpo_model",
            learning_rate=1e-6,
            num_generations=group_size,
            max_prompt_length=256,
            max_completion_length=512,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            report_to="none",
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
        )

        sample_prompts = [
            {"prompt": "Calculate the compound interest on 1000 at 10% for 2 years.", "answer": "210"},
            {"prompt": "What is the penalty if a card is blocked immediately after theft?", "answer": "0"},
        ]
        dataset = Dataset.from_list(sample_prompts)

        trainer = GRPOTrainer(
            model=model_name,
            reward_funcs=[correctness_reward_func],
            args=training_args,
            train_dataset=dataset,
            peft_config=peft_config,
        )
        return trainer

    except (ImportError, AttributeError):
        print("[GRPO] Note: Install trl>=0.14 for native GRPOTrainer support.")
        return None


def main():
    parser = argparse.ArgumentParser(description="20. Group Relative Policy Optimization (GRPO)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model identifier")
    parser.add_argument("--output-dir", type=str, default="./outputs/grpo_model", help="Directory to save model")
    parser.add_argument("--group-size", type=int, default=4, help="Group size G (completions per prompt)")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default: dry-run inspection)")
    args = parser.parse_args()

    print("=" * 75)
    print("20. GROUP RELATIVE POLICY OPTIMIZATION (GRPO) — STANDALONE MODULE")
    print("=" * 75)
    print(f"• Base Model:       {args.model}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Group Size (G):   {args.group_size} completions per prompt")
    print(f"• Critic Model:     NONE (Advantage calculated via group baseline)")
    print(f"• DeepSeek-R1 Math: A_i = (r_i - mean(R)) / (std(R) + eps)")
    print(f"• VRAM Savings:     ~50% memory reduction vs classical PPO")

    # Verify group normalization math
    mock_rewards = torch.tensor([[1.0, 0.0, 0.5, 1.5]])
    advantages = compute_group_relative_advantages(mock_rewards)
    print(f"\n✓ Group Advantage Demo: Rewards {mock_rewards.tolist()} -> Advantages {advantages.tolist()}")

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
        print(f"      python fine_tuning_techniques/20_grpo.py --train --model {args.model}")
        print("=" * 75)
        return

    print("\n🚀 Starting GRPO training execution...")
    trainer = setup_grpo_framework(model_name=args.model, group_size=args.group_size)
    if trainer is not None:
        trainer.train()
        trainer.save_model(args.output_dir)
        print(f"\n✓ GRPO model saved successfully to: {args.output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
