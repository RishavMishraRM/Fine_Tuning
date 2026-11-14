"""
09. Task-Adaptive Pretraining (TAPT)
=====================================
Also from Gururangan et al. (ACL 2020).

Concept:
- While DAPT pre-trains on the broad domain (e.g., all biomedical literature), TAPT
  focuses narrowly on the exact unlabeled task dataset distribution (e.g., only the
  unlabeled customer inquiries of your specific application).
- Surprisingly, pretraining on small, hyper-targeted task datasets (even just a few thousand
  unlabeled examples) often yields higher performance gains than pretraining on broad domain data.

Recommended Progressive Pipeline:
    Base Foundation LLM
             │
             ▼
    DAPT (Broad Domain Corpus: e.g. 10M-1B tokens)
             │
             ▼
    TAPT (Task-Specific Unlabeled Corpus: e.g. 50k customer tickets)
             │
             ▼
    Task Fine-Tuning (Labeled SFT on 1k-10k prompt/response pairs)
"""

from typing import List
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


def load_unlabeled_task_corpus() -> Dataset:
    """Unlabeled queries drawn directly from the production task environment."""
    task_queries = [
        {"text": "Customer ticket: Need immediate assistance with an unrecognized ATM deduction."},
        {"text": "Customer ticket: How do I enable international online transactions on my debit card?"},
        {"text": "Customer ticket: Foreclosure letter generation error on personal loan account."},
        {"text": "Customer ticket: Refund initiated by merchant on UPI transaction has not reflected."},
    ]
    return Dataset.from_list(task_queries)


def setup_tapt_pipeline(model_name: str = "mistralai/Mistral-7B-v0.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    task_corpus = load_unlabeled_task_corpus()

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
        )

    tokenized_dataset = task_corpus.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    training_args = TrainingArguments(
        output_dir="./outputs/tapt_task_model",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,  # Very low learning rate to tune strictly to task distribution
        num_train_epochs=3,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=default_data_collator,
    )
    return trainer


if __name__ == "__main__":
    print("Task-Adaptive Pretraining (TAPT) module ready.")
    print("Pre-trains on unlabeled task queries immediately before supervised fine-tuning.")
