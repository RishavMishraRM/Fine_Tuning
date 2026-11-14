"""
14. Prefix Tuning
==================
Introduced by Li & Liang (ACL 2021).

Concept:
- Instead of adding parameters inside the feed-forward networks (as in Adapters),
  Prefix Tuning prepends learnable, task-specific continuous virtual Key and Value
  vectors (P_K, P_V) directly to the Multi-Head Self-Attention in EVERY transformer layer.
- At layer l, for key sequence K_l and value sequence V_l:
      Keys   = [ P_K^(l) ; K_l ]
      Values = [ P_V^(l) ; V_l ]
      Attention(Q_l, Keys, Values)

Key Nuance:
To stabilize training, the continuous prefixes P_K, P_V are initially parameterized via
a Multi-Layer Perceptron (MLP): P = MLP(e_prefix). Once trained, only the resulting
prefix tensors are stored.

Parameter count: ~0.1% of base model.
"""

from typing import Dict, Any
import torch
from datasets import Dataset
from peft import (
    PrefixTuningConfig,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


def setup_prefix_tuning_pipeline(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    num_virtual_tokens: int = 30,
):
    """Configures Prefix Tuning across all transformer attention layers."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=num_virtual_tokens,  # Length of prefix prepended to Key/Value projections
        prefix_projection=True,  # Uses MLP reparameterization for optimization stability
    )

    prefix_model = get_peft_model(base_model, peft_config)
    prefix_model.print_trainable_parameters()

    sample_data = [
        {"text": "Task: Summarize policy. Input: The term deposit matures in 12 months. Output: 1-year deposit tenure."},
    ]
    dataset = Dataset.from_list(sample_data)

    def tokenize_fn(x):
        return tokenizer(x["text"], truncation=True, max_length=256, padding="max_length")

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])

    training_args = TrainingArguments(
        output_dir="./outputs/prefix_tuning_model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-3,  # Prefix Tuning often benefits from higher learning rate than full fine-tuning
        num_train_epochs=5,
        bf16=True,
        logging_steps=5,
        report_to="none",
    )

    trainer = Trainer(
        model=prefix_model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=default_data_collator,
    )
    return trainer


if __name__ == "__main__":
    print("Prefix Tuning pipeline initialized.")
    print("Prepends virtual Key and Value projections across all multi-head attention layers.")
