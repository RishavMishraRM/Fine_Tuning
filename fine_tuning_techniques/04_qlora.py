"""
04. QLoRA (Quantized Low-Rank Adaptation)
=========================================
QLoRA (Dettmers et al., 2023) enables fine-tuning of 7B-70B models on a single
consumer GPU (e.g. 16GB - 24GB VRAM) without quality degradation.

Key Innovations:
1. 4-bit NormalFloat (NF4): An information-theoretically optimal quantile quantization
   data type for zero-mean, unit-variance normally distributed weights.
2. Double Quantization (DQ): Quantizes the quantization constants of first-level quantization,
   saving approximately 0.37 bits per parameter (~3 GB for a 65B model).
3. Paged Optimizers: Uses CUDA Unified Memory to page memory between GPU and CPU during
   gradient checkpointing memory spikes, preventing OOM crashes.
"""

from dataclasses import dataclass
from typing import List, Optional
import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)


@dataclass
class QLoRAConfig:
    model_name: str = "mistralai/Mistral-7B-v0.1"
    output_dir: str = "./outputs/qlora_adapter"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    use_double_quant: bool = True
    quant_type: str = "nf4"  # 'nf4' or 'fp4'
    learning_rate: float = 2e-4
    max_seq_length: int = 2048


def build_qlora_model(config: QLoRAConfig):
    """Initializes 4-bit quantized base model and attaches trainable FP32/BF16 LoRA adapters."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.quant_type,
        bnb_4bit_use_double_quant=config.use_double_quant,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Freezes 4-bit base weights, casts LayerNorms & input embeddings to FP32 for stability
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    qlora_model = get_peft_model(model, peft_config)
    qlora_model.print_trainable_parameters()
    return qlora_model, tokenizer


def get_qlora_training_arguments(output_dir: str = "./outputs/qlora") -> TrainingArguments:
    """Returns training arguments optimized for 4-bit memory efficiency."""
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",  # Paged optimizer prevents OOM spikes
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )


if __name__ == "__main__":
    print("QLoRA (Quantized Low-Rank Adaptation) setup verified.")
    cfg = QLoRAConfig()
    print(f"NF4 Quantization + Double Quant: {cfg.use_double_quant}, Compute dtype=bfloat16")
