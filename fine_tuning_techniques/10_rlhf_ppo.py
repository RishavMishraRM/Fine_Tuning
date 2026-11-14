"""
10. Reinforcement Learning from Human Feedback (RLHF)
======================================================
RLHF (Ouyang et al., 2022 / InstructGPT) aligns models with human values, safety,
and helpfulness through a 3-stage pipeline:

Stage 1: Supervised Fine-Tuning (SFT) on curated demonstration data.
Stage 2: Reward Model (RM) Training:
         A model r_psi(x, y) trained on pairwise comparison data (y_w preferred over y_l):
         L_RM = - E_{(x, y_w, y_l)} [ log sigma( r_psi(x, y_w) - r_psi(x, y_l) ) ]
Stage 3: Reinforcement Learning (PPO) against the learned reward model:
         Objective: max_theta E_{(x, y) ~ pi_theta} [ r_psi(x, y) - beta * D_KL(pi_theta(y|x) || pi_ref(y|x)) ]
         The KL penalty ensures the policy does not collapse into reward gaming/hacking.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
)


@dataclass
class RLHFStage2RewardModelConfig:
    model_name: str = "mistralai/Mistral-7B-v0.1"
    num_labels: int = 1  # Scalar reward output


def build_reward_model(model_name: str = "mistralai/Mistral-7B-v0.1"):
    """
    Stage 2: Loads or trains a SequenceClassification model with 1 output head
    that outputs scalar score r(x, y).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Scalar reward prediction head (classification head with 1 linear output)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return reward_model, tokenizer


def setup_rlhf_ppo_trainer(
    policy_model_name: str = "mistralai/Mistral-7B-v0.1",
    reward_model_name: str = "mistralai/Mistral-7B-v0.1",
):
    """
    Stage 3: Sets up PPO Trainer with Actor (Policy), Reference Model (Frozen SFT),
    Value Head (Critic), and Reward Model.
    """
    ppo_config = PPOConfig(
        model_name=policy_model_name,
        learning_rate=1.4e-5,
        batch_size=8,
        mini_batch_size=2,
        gradient_accumulation_steps=4,
        init_kl_coef=0.2,  # Beta: KL divergence penalty coefficient
        target_kl=0.1,
        adap_kl_ctrl=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Actor policy with integrated linear Value Head for Generalized Advantage Estimation (GAE)
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        policy_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Reference policy is kept frozen to compute KL penalty
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        policy_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )
    return ppo_trainer


if __name__ == "__main__":
    print("RLHF (Reinforcement Learning from Human Feedback) 3-Stage Pipeline configured.")
    print("Stage 1: SFT -> Stage 2: Reward Model (Bradley-Terry) -> Stage 3: PPO with KL Penalty.")
