"""
05. Supervised Fine-Tuning (SFT)
=================================
Supervised Fine-Tuning adapts a base (pre-trained) causal language model to follow
user instructions or conduct dialogue using curated pairs of inputs and target outputs:
    Loss = - sum_{t=1}^{T} log P(y_t | x, y_{<t}; theta)

In standard SFT:
- Tokens in the prompt x are often masked (loss calculated only on completion tokens y).
- TRL's SFTTrainer manages prompt formatting, packing sequences to max context length,
  and computing cross-entropy loss over response tokens.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


@dataclass
class SFTConfig:
    model_name: str = "mistralai/Mistral-7B-v0.1"
    output_dir: str = "./outputs/sft_model"
    max_seq_length: int = 1024
    learning_rate: float = 2e-5
    batch_size: int = 4
    num_train_epochs: int = 2


def build_sample_sft_dataset() -> Dataset:
    """Creates a sample multi-turn / instruction-response dataset."""
    samples = [
        {
            "messages": [
                {"role": "system", "content": "You are a professional assistant."},
                {"role": "user", "content": "Explain compound interest."},
                {"role": "assistant", "content": "Compound interest is interest calculated on initial principal plus accumulated interest."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a professional assistant."},
                {"role": "user", "content": "What is an index fund?"},
                {"role": "assistant", "content": "An index fund is a portfolio constructed to match or track components of a financial market index."},
            ]
        },
    ]
    return Dataset.from_list(samples)


def setup_sft_trainer(config: SFTConfig):
    """Sets up TRL SFTTrainer with chat template formatting."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    dataset = build_sample_sft_dataset()

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        max_seq_length=config.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )
    return trainer


if __name__ == "__main__":
    print("Supervised Fine-Tuning (SFT) module initialized.")
    dataset = build_sample_sft_dataset()
    print(f"Sample dataset size: {len(dataset)} dialog turns.")
