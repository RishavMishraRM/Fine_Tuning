"""
11. Direct Preference Optimization (DPO)
=========================================
Introduced by Rafailov et al. (NeurIPS 2023).

Breakthrough Insight:
RLHF requires training a separate reward model r(x, y) and performing unstable,
memory-intensive PPO reinforcement learning with 4 concurrent models (Actor, Critic,
Reward Model, Reference Model).

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

from typing import Dict, List
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer


def build_preference_dataset() -> Dataset:
    """Builds pairwise dataset containing prompt, chosen (preferred), and rejected responses."""
    data = [
        {
            "prompt": "Customer: My card was stolen. What should I do?",
            "chosen": "Immediately freeze your card via the bank app or official hotline. Do not disclose your PIN or OTP to anyone.",
            "rejected": "Tell me your 16-digit card number and PIN so I can look into your account and cancel it for you.",
        },
        {
            "prompt": "Customer: When will my refund arrive?",
            "chosen": "Refund timelines depend on merchant settlement and banking clearing cycles. Please verify the refund ARN reference with your merchant.",
            "rejected": "Your refund is guaranteed to arrive in your account by tomorrow morning.",
        },
    ]
    return Dataset.from_list(data)


def setup_dpo_trainer(model_name: str = "mistralai/Mistral-7B-v0.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Optional PEFT LoRA adapter setup for parameter-efficient DPO
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
        learning_rate=5e-6,  # DPO requires small learning rate (5e-7 to 1e-5)
        num_train_epochs=3,
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # When None, DPOTrainer automatically creates a frozen reference copy
        args=training_args,
        beta=0.1,  # Beta: Implicit reward margin scaling factor (typically 0.1 - 0.5)
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_length=512,
        max_prompt_length=256,
    )
    return dpo_trainer


if __name__ == "__main__":
    print("Direct Preference Optimization (DPO) configured.")
    print("Directly aligns policy on (prompt, chosen, rejected) tuples without a reward model.")
