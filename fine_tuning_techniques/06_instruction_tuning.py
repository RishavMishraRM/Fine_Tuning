"""
06. Instruction Fine-Tuning
============================
Instruction Tuning trains a model to generalize to unseen tasks specified via
natural language prompts.

Critical Mechanism: Response-Only Loss Masking
- If we compute cross-entropy loss over the entire sequence (Prompt + Response), the model
  wastes capacity learning to generate the prompt text.
- With response-only masking (via TRL's DataCollatorForCompletionOnlyLM), prompt tokens are
  assigned label -100 (ignored by PyTorch cross_entropy), so gradients flow EXCLUSIVELY
  through the generated response tokens.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM


ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{response}"""


def format_instruction_sample(sample: Dict[str, str]) -> str:
    """Formats raw fields into standard instruction-prompt structure."""
    return ALPACA_TEMPLATE.format(
        instruction=sample.get("instruction", ""),
        input_text=sample.get("input", ""),
        response=sample.get("response", ""),
    )


def build_instruction_dataset(tokenizer) -> Dataset:
    raw_data = [
        {
            "instruction": "Classify the sentiment of the banking review.",
            "input": "My refund was credited within two hours. Excellent service!",
            "response": "Positive",
        },
        {
            "instruction": "Identify the primary intent.",
            "input": "I lost my credit card during travel.",
            "response": "Credit Card Emergency / Card Block",
        },
    ]

    formatted_texts = [format_instruction_sample(x) for x in raw_data]
    return Dataset.from_list([{"text": text} for text in formatted_texts])


def setup_instruction_trainer(model_name: str = "mistralai/Mistral-7B-v0.1"):
    """Configures Trainer with completion-only response loss masking."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Target delimiter marking where response begins
    response_template = "### Response:\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    dataset = build_instruction_dataset(tokenizer)

    def tokenize_fn(example):
        return tokenizer(example["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_fn, batched=False, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    training_args = TrainingArguments(
        output_dir="./outputs/instruction_model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        bf16=True,
        logging_steps=5,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
    )
    return trainer


if __name__ == "__main__":
    print("Instruction Fine-Tuning pipeline with Response-Only Masking ready.")
    sample = {
        "instruction": "Extract the currency code.",
        "input": "Transfer 500 EUR to account X.",
        "response": "EUR",
    }
    print("\nSample formatted prompt:")
    print(format_instruction_sample(sample))
