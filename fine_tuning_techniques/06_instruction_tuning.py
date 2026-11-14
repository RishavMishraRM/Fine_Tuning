# ==============================================================================
# 06. Instruction Fine-Tuning (Masked Loss) — Standalone Fine-Tuning Script
# ==============================================================================
# Requirements:
#   pip install torch transformers trl datasets accelerate
#
# Usage:
#   # Dry-run inspection (default):
#   python fine_tuning_techniques/06_instruction_tuning.py
#
#   # Live training on GPU:
#   python fine_tuning_techniques/06_instruction_tuning.py --train --model Qwen/Qwen2.5-0.5B
# ==============================================================================
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

import argparse
from typing import Dict, Any, List


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "trl", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


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


def build_instruction_dataset(tokenizer):
    from datasets import Dataset

    raw_data = [
        {
            "instruction": "Classify the sentiment of the review.",
            "input": "My issue was resolved within two hours. Excellent service!",
            "response": "Positive",
        },
        {
            "instruction": "Identify the primary intent.",
            "input": "I lost my payment card during travel.",
            "response": "Card Emergency / Card Freeze",
        },
        {
            "instruction": "Summarize the technical mechanism.",
            "input": "DataCollatorForCompletionOnlyLM assigns label -100 to prompt tokens.",
            "response": "Prompt tokens are excluded from cross-entropy gradient calculations.",
        },
    ]

    formatted_texts = [format_instruction_sample(x) for x in raw_data]
    return Dataset.from_list([{"text": text} for text in formatted_texts])


def setup_instruction_trainer(model_name: str = "mistralai/Mistral-7B-v0.1"):
    """Configures Trainer with completion-only response loss masking."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from trl import DataCollatorForCompletionOnlyLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )

    training_args = TrainingArguments(
        output_dir="./outputs/instruction_model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
    )
    return trainer


def main():
    parser = argparse.ArgumentParser(description="06. Instruction Fine-Tuning")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model identifier")
    parser.add_argument("--output-dir", type=str, default="./outputs/instruction_model", help="Directory to save model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default: dry-run inspection)")
    args = parser.parse_args()

    print("=" * 75)
    print("06. INSTRUCTION FINE-TUNING (RESPONSE-ONLY MASKING) — STANDALONE MODULE")
    print("=" * 75)
    print(f"• Base Model:       {args.model}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Masking Engine:   DataCollatorForCompletionOnlyLM (label -100 on prompt)")
    print(f"• Template Style:   Alpaca (Instruction / Input / Response)")
    print(f"• VRAM Requirement: 16 - 24 GB for 7B models")

    missing = check_prerequisites()
    if missing:
        print("\n⚠️  MISSING PREREQUISITES:")
        print(f"  This script requires: {', '.join(missing)}")
        print("\n  To install the required packages:")
        print("      pip install torch transformers trl datasets accelerate")
        print("=" * 75)
        return

    if not args.train:
        print("\n💡 Mode: DRY-RUN / INSPECTION ONLY")
        print("  All dependencies are verified! To execute live training on GPU:")
        print(f"      python fine_tuning_techniques/06_instruction_tuning.py --train --model {args.model}")
        print("=" * 75)
        return

    print("\n🚀 Starting Instruction Fine-Tuning execution...")
    trainer = setup_instruction_trainer(model_name=args.model)
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\n✓ Instruction-tuned model saved successfully to: {args.output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
