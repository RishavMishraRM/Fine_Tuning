"""
17. ORPO (Odds Ratio Preference Optimization)
==============================================
Introduced by Hong, Lee, and Thorne (2024).

The Paradigm Shift:
Both RLHF and DPO require an initial Supervised Fine-Tuning (SFT) phase, followed
by preference optimization against a separate reference model (pi_ref).

ORPO achieves SFT and Preference Alignment SIMULTANEOUSLY in a single training stage,
without requiring a reference model:
    L_{ORPO} = L_{SFT} + lambda * L_{OR}

Where the Odds Ratio (OR) measures how much more likely the model generates the
chosen response y_w compared to the rejected response y_l:
    odds_theta(y|x) = P_theta(y|x) / (1 - P_theta(y|x))
    L_{OR} = - log sigma( log( odds_theta(y_w|x) / odds_theta(y_l|x) ) )

Key Advantages:
- Single-step training: Eliminates the 2-step SFT -> DPO pipeline.
- Memory efficiency: Does NOT load or maintain a frozen reference model.
- Prevents degradation of cross-entropy generation quality while penalizing rejected tokens.
"""

from dataclasses import dataclass
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ORPOConfig, ORPOTrainer


def build_orpo_dataset() -> Dataset:
    """Builds pairwise dataset formatted with prompt, chosen, and rejected fields."""
    samples = [
        {
            "prompt": "How do I report an unrecognized payment on my bank statement?",
            "chosen": "Contact your bank's fraud line immediately or lock your card through the official banking app. Note the transaction ID.",
            "rejected": "Don't worry about it, unrecognized debits always resolve themselves automatically after a week.",
        },
        {
            "prompt": "Can the assistant waive my credit card annual fee?",
            "chosen": "As an AI, I cannot directly alter fees on your account. You can request a fee review by contacting the card issuer's billing team.",
            "rejected": "Yes, I have officially waived your credit card annual fee. Your next bill will reflect this.",
        },
    ]
    return Dataset.from_list(samples)


def setup_orpo_trainer(model_name: str = "mistralai/Mistral-7B-v0.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    dataset = build_orpo_dataset()

    orpo_args = ORPOConfig(
        output_dir="./outputs/orpo_model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=8e-6,
        beta=0.1,  # Lambda / Beta weight for the Odds Ratio penalty
        max_prompt_length=256,
        max_length=512,
        num_train_epochs=3,
        bf16=True,
        logging_steps=5,
        report_to="none",
    )

    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    return trainer


if __name__ == "__main__":
    print("ORPO (Odds Ratio Preference Optimization) module configured.")
    print("Simultaneous SFT + Preference Alignment without a reference model.")
