"""
08. Domain-Adaptive Pretraining (DAPT)
=======================================
Introduced by Gururangan et al. (ACL 2020) in 'Don't Stop Pretraining: Adapt Language
Models to Domains and Tasks'.

Concept:
- General web-scale pre-training does not capture domain-specific terminology,
  distribution shifts, and specialized syntax (e.g. BioMed, Legal, Finance).
- DAPT performs continued causal/masked pretraining on tens of millions of tokens from
  the target domain prior to task-specific fine-tuning.
- Metric to monitor: Domain Perplexity (PPL = exp(CrossEntropyLoss)). A steep drop in
  domain PPL indicates the model has internalized domain vocabulary and distribution.
"""

import math
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


def load_domain_corpus() -> Dataset:
    """Specialized domain text (e.g. medical clinical notes, statutory tax law, financial filings)."""
    domain_texts = [
        {"text": "Under Section 80C of the Income Tax Act, deductions up to 1.5 lakh INR are permissible."},
        {"text": "The reserve asset ratio must satisfy statutory liquidity ratio (SLR) requirements."},
        {"text": "Non-performing assets (NPA) are classified as sub-standard when overdue exceeds 90 days."},
    ]
    return Dataset.from_list(domain_texts)


def evaluate_domain_perplexity(trainer: Trainer) -> float:
    """Computes perplexity on held-out domain evaluation samples."""
    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss", 0.0)
    perplexity = math.exp(eval_loss)
    print(f"[DAPT] Domain Evaluation Loss: {eval_loss:.4f} || Perplexity: {perplexity:.2f}")
    return perplexity


def setup_dapt_pipeline(model_name: str = "mistralai/Mistral-7B-v0.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_domain_corpus()

    def tokenize_fn(x):
        return tokenizer(x["text"], truncation=True, max_length=512, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])

    # 80/20 train/eval split for domain perplexity tracking
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    training_args = TrainingArguments(
        output_dir="./outputs/dapt_domain_model",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=default_data_collator,
    )
    return trainer


if __name__ == "__main__":
    print("Domain-Adaptive Pretraining (DAPT) pipeline configured.")
    print("DAPT adapts general model weights to domain-specific syntax and vocabulary.")
