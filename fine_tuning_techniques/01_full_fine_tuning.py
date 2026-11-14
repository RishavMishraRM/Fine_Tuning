"""
01. Full Fine-Tuning (FFT)
===========================
Full Fine-Tuning updates all parameters (weights and biases) of the neural
network during backpropagation:
    W_new = W_old - lr * grad(L)

Key Characteristics:
- High compute & memory cost: Requires ~16 bytes of VRAM per parameter (weights in fp16/bf16,
  gradients in fp16/bf16, and optimizer states in fp32 for AdamW: 2 + 2 + 12 = 16 bytes).
  A 7B parameter model requires ~112 GB VRAM for training without offloading.
- Highest ceiling for capability modification and domain absorption.
- High risk of catastrophic forgetting of out-of-domain general knowledge.

When to use:
- Foundational domain shifts (e.g. teaching a general model medical chemistry from scratch).
- Multi-GPU compute clusters (e.g., 4x or 8x A100/H100) are available.
"""

from dataclasses import dataclass
from typing import Optional
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


@dataclass
class FFTConfig:
    model_name: str = "mistralai/Mistral-7B-v0.1"
    output_dir: str = "./outputs/fft_model"
    learning_rate: float = 2e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    bf16: bool = True
    gradient_checkpointing: bool = True


def prepare_sample_dataset(tokenizer, max_length: int = 512):
    """Generates a tokenized causal LM dataset for demonstration."""
    raw_data = [
        {"text": "Customer: What is your loan foreclosure policy? Assistant: Foreclosure depends on loan terms."},
        {"text": "Customer: I lost my credit card. Assistant: Immediately freeze your card via the mobile app."},
        {"text": "Customer: How do I verify a refund? Assistant: Check your transaction reference with the merchant."},
    ]
    dataset = Dataset.from_list(raw_data)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


def build_fft_trainer(config: Optional[FFTConfig] = None) -> Trainer:
    """Builds a Full Fine-Tuning Trainer instance."""
    config = config or FFTConfig()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with all parameters trainable (requires_grad=True by default)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Verify all parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[FFT] Trainable parameters: {trainable_params:,} / {total_params:,} (100.0%)")

    tokenized_dataset = prepare_sample_dataset(tokenizer, config.max_seq_length)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        bf16=config.bf16,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        save_total_limit=2,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    return trainer


if __name__ == "__main__":
    print("Full Fine-Tuning (FFT) Pipeline configured.")
    print("Run `trainer.train()` on a multi-GPU environment to execute full parameter backprop.")
