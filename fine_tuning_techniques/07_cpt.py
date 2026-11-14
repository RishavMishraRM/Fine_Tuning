"""
07. Continued Pretraining (CPT)
================================
Continued Pretraining (also known as second-stage pretraining) trains a completed
foundation model on a massive corpus of raw, unstructured text using the original
unsupervised next-token prediction objective:
    L_{CPT} = - sum_{i=1}^{N} log P(w_i | w_1, ..., w_{i-1}; theta)

Key Differences from Supervised Tuning:
- Uses raw, unformatted text without instruction/response delimiters.
- Employs token concatenation and packing (chunking continuous streams into block_size=2048/4096)
  to ensure maximum attention efficiency without padding waste.
- Uses a significantly smaller learning rate (e.g., 1e-5 to 5e-5) with a cosine decay
  schedule to avoid catastrophic degradation of initial weights.
"""

from itertools import chain
from typing import Dict, List
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


def build_raw_corpus() -> Dataset:
    """Simulates a large raw text corpus (e.g. corporate filings, books, code repositories)."""
    documents = [
        {"text": "The Reserve Bank of India governs monetary policy and regulates commercial banks."},
        {"text": "Liquidity adjustment facilities include repo rate and reverse repo rate operations."},
        {"text": "Capital adequacy ratio ensures that financial institutions absorb reasonable losses."},
    ]
    return Dataset.from_list(documents)


def group_texts(examples: Dict[str, List[int]], block_size: int = 512) -> Dict[str, List[List[int]]]:
    """
    Concatenates continuous tokens from all documents and splits them into
    uniform block_size chunks. Essential for maximum compute efficiency in pretraining.
    """
    concatenated = list(chain(*examples["input_ids"]))
    total_length = len(concatenated)

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    chunks = [
        concatenated[i : i + block_size]
        for i in range(0, total_length, block_size)
    ]

    return {
        "input_ids": chunks,
        "labels": [chunk.copy() for chunk in chunks],  # Next-token prediction labels
    }


def setup_cpt_trainer(model_name: str = "mistralai/Mistral-7B-v0.1", block_size: int = 512):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = build_raw_corpus()

    # Step 1: Tokenize raw text
    def tokenize_fn(example):
        return tokenizer(example["text"])

    tokenized = raw_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Step 2: Pack into continuous block_size sequences
    lm_dataset = tokenized.map(
        lambda x: group_texts(x, block_size=block_size),
        batched=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    training_args = TrainingArguments(
        output_dir="./outputs/cpt_model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=3e-5,  # Conservative learning rate to preserve foundation reasoning
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=default_data_collator,
    )
    return trainer


if __name__ == "__main__":
    print("Continued Pretraining (CPT) pipeline initialized.")
    print("Packed chunk sequence formatting demonstrated.")
