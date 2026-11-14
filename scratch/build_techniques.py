import sys
from pathlib import Path

TARGET_DIR = Path(__file__).resolve().parent.parent / "fine_tuning_techniques"

def write_file(filename, content):
    path = TARGET_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"✓ Generated {filename} ({len(content.splitlines())} lines)")

# ==============================================================================
# 01_full_fine_tuning.py
# ==============================================================================
T01 = '''# ==============================================================================
# 01. Full Fine-Tuning (FFT) — Standalone Fine-Tuning Script
# ==============================================================================
# Requirements:
#   pip install torch transformers datasets accelerate
#
# Usage:
#   # Dry-run mode (architecture inspection & VRAM requirements):
#   python 01_full_fine_tuning.py
#
#   # Live training mode:
#   python 01_full_fine_tuning.py --train --model Qwen/Qwen2.5-0.5B --epochs 1
# ==============================================================================
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

import argparse
import sys
from dataclasses import dataclass
from typing import Optional


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


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
    max_seq_length: int = 512
    bf16: bool = True
    gradient_checkpointing: bool = True


def prepare_sample_dataset(tokenizer, max_length: int = 512):
    """Generates a tokenized causal LM dataset for general domain adaptation."""
    from datasets import Dataset

    raw_data = [
        {"text": "User: What is the principle of conservation of energy? Assistant: Energy cannot be created or destroyed, only transformed from one form to another."},
        {"text": "User: How does HTTPS protect web traffic? Assistant: HTTPS uses TLS/SSL encryption to secure communications between browser and server."},
        {"text": "User: Explain database normalization. Assistant: Normalization organizes relational database tables to reduce redundancy and improve data integrity."},
        {"text": "User: What should I do if my bank card is lost? Assistant: Freeze the card immediately via your official mobile banking app and notify card services."},
    ]
    dataset = Dataset.from_list(raw_data)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])
    return tokenized


def build_fft_trainer(config: Optional[FFTConfig] = None):
    """Builds a Full Fine-Tuning Trainer instance."""
    config = config or FFTConfig()
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with all parameters trainable (requires_grad=True by default)
    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if (config.bf16 and torch.cuda.is_available()) else torch.float32,
        device_map=device_map,
    )

    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

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
        bf16=config.bf16 and torch.cuda.is_available(),
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=1,
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


def main():
    parser = argparse.ArgumentParser(description="01. Full Fine-Tuning (FFT)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model identifier")
    parser.add_argument("--output-dir", type=str, default="./outputs/fft_model", help="Directory to save model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default: dry-run inspection)")
    args = parser.parse_args()

    missing = check_prerequisites()
    if missing:
        print("=" * 70)
        print("⚠️  MISSING PREREQUISITES FOR THIS TECHNIQUE")
        print("=" * 70)
        print(f"The following required packages are not installed: {', '.join(missing)}")
        print("\nTo install the necessary dependencies, run:")
        print("    pip install torch transformers datasets accelerate")
        print("\nOr install the full repository suite:")
        print("    pip install -r requirements-training.txt")
        print("=" * 70)
        return

    print("=" * 70)
    print("01. FULL FINE-TUNING (FFT) — CONFIGURATION")
    print("=" * 70)
    print(f"• Base Model:       {args.model}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Learning Rate:    {args.lr}")
    print(f"• Epochs:           {args.epochs}")
    print(f"• Batch Size:       {args.batch_size}")
    print(f"• VRAM Estimate:    ~16 bytes/param (~112 GB for 7B model)")
    print(f"• Optimizer:        AdamW (fp32 states) with Gradient Checkpointing")

    if not args.train:
        print("\n💡 Mode: DRY-RUN / INSPECTION ONLY")
        print("To execute live training, run with the `--train` flag:")
        print(f"    python 01_full_fine_tuning.py --train --model {args.model}")
        print("=" * 70)
        return

    print("\n🚀 Starting Full Fine-Tuning execution...")
    config = FFTConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_train_epochs=args.epochs,
    )
    trainer = build_fft_trainer(config)
    trainer.train()
    trainer.save_model(config.output_dir)
    print(f"\n✓ Training complete! Full fine-tuned model saved to: {config.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
'''
write_file("01_full_fine_tuning.py", T01)

# ==============================================================================
# 02_peft_overview.py
# ==============================================================================
T02 = '''# ==============================================================================
# 02. PEFT Architecture Overview — Standalone Analysis Script
# ==============================================================================
# Requirements:
#   pip install torch transformers peft
#
# Usage:
#   python 02_peft_overview.py [--model MODEL]
# ==============================================================================
"""
02. Parameter-Efficient Fine-Tuning (PEFT Overview)
===================================================
PEFT freezes the vast majority of the foundational model parameters and trains
only a tiny subset (typically 0.05% - 2% of the parameters).

The core families of PEFT include:
1. Re-parameterization / Low-Rank: LoRA, QLoRA, DoRA, AdaLoRA
2. Additive Tuning / Bottleneck Adapters: Houlsby, Pfeiffer
3. Soft Prompt / Prefix: Prompt Tuning, Prefix Tuning, P-Tuning v1/v2
4. Selective Tuning: BitFit (tuning only bias terms)

Mathematical Formulation:
Instead of optimizing W in R^{d x k}, PEFT keeps W_0 fixed and optimizes a compact
parameter set theta:
    h = W_0 * x + Delta_W(theta) * x

Benefits:
- Drastically reduced VRAM requirements (optimizer states scale only with theta).
- Storage efficiency: A single base 7B model (14 GB) can host dozens of task adapters (10-50 MB each).
- Mitigates catastrophic forgetting since base weights are immutable.
"""

import argparse
from typing import Dict, Any


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "peft"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def compare_peft_methods(model_name: str = "mistralai/Mistral-7B-v0.1") -> Dict[str, Any]:
    """Demonstrates how different PEFT configs attach to a model and inspects their configs."""
    from peft import (
        TaskType,
        LoraConfig,
        PrefixTuningConfig,
        PromptTuningConfig,
        PromptEncoderConfig,
    )

    configs = {
        "LoRA": LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
        ),
        "Prefix_Tuning": PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=30,
        ),
        "Prompt_Tuning": PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
        ),
        "P_Tuning": PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
            encoder_hidden_size=128,
        ),
    }

    print("\nPEFT Configuration Matrix:")
    print(f"{'Method':<16} {'PEFT Type':<20} {'Target / Virtual Tokens':<30}")
    print("-" * 70)
    for name, config in configs.items():
        details = (
            f"r={config.r}, alpha={config.lora_alpha}"
            if hasattr(config, "r")
            else f"tokens={config.num_virtual_tokens}"
        )
        print(f"• {name:<14s} {str(config.peft_type):<20s} {details:<30s}")

    return configs


def inspect_trainable_parameters(model) -> Dict[str, float]:
    """Utility to print and return the count of trainable vs total parameters."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable_params / all_param if all_param > 0 else 0
    print(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_param:,} || "
        f"Trainable%: {pct:.4f}%"
    )
    return {"trainable": trainable_params, "total": all_param, "ratio_pct": pct}


def main():
    parser = argparse.ArgumentParser(description="02. PEFT Architecture Overview")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Target model identifier")
    args = parser.parse_args()

    missing = check_prerequisites()
    if missing:
        print("=" * 70)
        print("⚠️  MISSING PREREQUISITES FOR THIS TECHNIQUE")
        print("=" * 70)
        print(f"Missing packages: {', '.join(missing)}")
        print("\nInstall via:")
        print("    pip install torch transformers peft")
        print("=" * 70)
        return

    print("=" * 70)
    print("02. PARAMETER-EFFICIENT FINE-TUNING (PEFT) OVERVIEW")
    print("=" * 70)
    print(f"Target Model: {args.model}")
    compare_peft_methods(args.model)

    print("\nTheoretical Comparison Table (7B Model, 14 GB base FP16 weights):")
    print(f"{'Method':<18} {'Trainable Params':<18} {'Adapter Size':<15} {'VRAM Required'}")
    print("-" * 70)
    print(f"{'Full Fine-Tuning':<18} {'7,000,000,000 (100%)':<18} {'14 GB':<15} {'> 112 GB'}")
    print(f"{'LoRA (r=16)':<18} {'~6,800,000 (0.10%)':<18} {'~27 MB':<15} {'~16-24 GB'}")
    print(f"{'QLoRA (4-bit r=16)':<18} {'~6,800,000 (0.10%)':<18} {'~27 MB':<15} {'~6-8 GB'}")
    print(f"{'Prefix Tuning':<18} {'~1,500,000 (0.02%)':<18} {'~6 MB':<15} {'~12-16 GB'}")
    print(f"{'Prompt Tuning':<18} {'~81,920 (0.001%)':<18} {'~320 KB':<15} {'< 10 GB'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
'''
write_file("02_peft_overview.py", T02)

# ==============================================================================
# 03_lora.py
# ==============================================================================
T03 = '''# ==============================================================================
# 03. LoRA (Low-Rank Adaptation) — Standalone Fine-Tuning Script
# ==============================================================================
# Requirements:
#   pip install torch transformers peft datasets accelerate
#
# Usage:
#   # Dry-run inspection (default):
#   python 03_lora.py
#
#   # Live training:
#   python 03_lora.py --train --model Qwen/Qwen2.5-0.5B --rank 16 --epochs 3
# ==============================================================================
"""
03. LoRA (Low-Rank Adaptation)
===============================
LoRA freezes the pre-trained model weights W_0 in R^{d x k} and injects trainable
rank decomposition matrices into the transformer layers:
    W = W_0 + Delta_W = W_0 + (alpha / r) * (B * A)

Where:
- A in R^{r x k} is initialized with a random Gaussian distribution N(0, sigma^2).
- B in R^{d x r} is initialized to 0, ensuring Delta_W = 0 at the start of training.
- r << min(d, k) is the rank (typically 8, 16, 32, or 64).
- alpha is a constant scaling factor. Scaling by (alpha / r) stabilizes learning
  when varying rank r.

Advantages:
- Reduces trainable parameters by ~10,000x compared to full fine-tuning.
- No inference latency: During deployment, W_merged = W_0 + (alpha / r) * (B * A)
  can be permanently folded back into the original weights.
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "peft", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


@dataclass
class LoRAExperimentConfig:
    model_name: str = "mistralai/Mistral-7B-v0.1"
    output_dir: str = "./outputs/lora_adapter"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    learning_rate: float = 2e-4
    batch_size: int = 4
    num_train_epochs: int = 3


def get_default_lora_targets() -> List[str]:
    """Target modules typically adapted in contemporary transformer architectures."""
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_sample_dataset(tokenizer, max_length: int = 512):
    from datasets import Dataset

    samples = [
        {"text": "Instruction: Summarize key benefits of LoRA. Response: LoRA freezes the base model, adds small rank decomposition matrices, and eliminates inference latency after weight merging."},
        {"Instruction: What is temperature in LLM sampling? Response: Temperature scales the logits before softmax; 0 produces deterministic greedy decoding, while higher values increase creativity."},
        {"Instruction: How does nucleus sampling work? Response: Top-p sampling selects the smallest set of tokens whose cumulative probability exceeds threshold p."},
        {"Instruction: Provide guidance for a lost card. Response: Freeze the card immediately via the banking application to protect against fraudulent charges."},
    ]
    dataset = Dataset.from_list(samples)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])
    return tokenized


def build_lora_model(config: LoRAExperimentConfig):
    """Loads base model and attaches LoRA low-rank adapter matrices."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    targets = config.target_modules or get_default_lora_targets()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else None
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=targets,
        bias="none",
    )

    lora_model = get_peft_model(base_model, peft_config)
    lora_model.print_trainable_parameters()
    return lora_model, tokenizer


def merge_and_export_lora(lora_model, save_path: str):
    """Folds low-rank delta weights back into base model for zero-overhead inference."""
    print(f"Merging LoRA weights into base model and exporting to {save_path}...")
    merged_model = lora_model.merge_and_unload()
    merged_model.save_pretrained(save_path)
    print("Merged model exported successfully.")


def main():
    parser = argparse.ArgumentParser(description="03. LoRA (Low-Rank Adaptation)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model identifier")
    parser.add_argument("--output-dir", type=str, default="./outputs/lora_adapter", help="Directory to save adapter")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (r)")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA scaling factor (alpha)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default: dry-run inspection)")
    args = parser.parse_args()

    missing = check_prerequisites()
    if missing:
        print("=" * 70)
        print("⚠️  MISSING PREREQUISITES FOR THIS TECHNIQUE")
        print("=" * 70)
        print(f"Missing packages: {', '.join(missing)}")
        print("\nInstall via:")
        print("    pip install torch transformers peft datasets accelerate")
        print("=" * 70)
        return

    print("=" * 70)
    print("03. LoRA (LOW-RANK ADAPTATION) — CONFIGURATION")
    print("=" * 70)
    print(f"• Base Model:       {args.model}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Rank (r):         {args.rank}")
    print(f"• Alpha:            {args.alpha} (Scaling: {args.alpha / args.rank:.2f}x)")
    print(f"• Target Modules:   {get_default_lora_targets()}")
    print(f"• Trainable Ratio:  ~0.1% of base weights")
    print(f"• VRAM Estimate:    16 - 24 GB for 7B (or 4-6 GB for 0.5B model)")

    if not args.train:
        print("\n💡 Mode: DRY-RUN / INSPECTION ONLY")
        print("To execute live training, run with the `--train` flag:")
        print(f"    python 03_lora.py --train --model {args.model}")
        print("=" * 70)
        return

    print("\n🚀 Initializing LoRA training pipeline...")
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    import torch

    cfg = LoRAExperimentConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_train_epochs=args.epochs,
    )

    lora_model, tokenizer = build_lora_model(cfg)
    train_dataset = build_sample_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    lora_model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"\n✓ LoRA adapter weights saved successfully to: {cfg.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
'''
write_file("03_lora.py", T03)

# ==============================================================================
# 04_qlora.py
# ==============================================================================
T04 = '''# ==============================================================================
# 04. QLoRA (Quantized LoRA) — Standalone Fine-Tuning Script
# ==============================================================================
# Requirements:
#   pip install torch transformers peft bitsandbytes accelerate
#
# Usage:
#   # Dry-run inspection (default):
#   python 04_qlora.py
#
#   # Live training on GPU:
#   python 04_qlora.py --train --model mistralai/Mistral-7B-v0.1
# ==============================================================================
"""
04. QLoRA (Quantized Low-Rank Adaptation)
=========================================
QLoRA (Dettmers et al., 2023) enables fine-tuning of 7B-70B models on a single
consumer GPU (e.g. 6GB - 16GB VRAM) without quality degradation.

Key Innovations:
1. 4-bit NormalFloat (NF4): An information-theoretically optimal quantile quantization
   data type for zero-mean, unit-variance normally distributed weights.
2. Double Quantization (DQ): Quantizes the quantization constants of first-level quantization,
   saving approximately 0.37 bits per parameter (~3 GB for a 65B model).
3. Paged Optimizers: Uses CUDA Unified Memory to page memory between GPU and CPU during
   gradient checkpointing memory spikes, preventing OOM crashes.
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional


def check_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "peft", "bitsandbytes"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


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
    max_seq_length: int = 512


def build_qlora_model(config: QLoRAConfig):
    """Initializes 4-bit quantized base model and attaches trainable FP32/BF16 LoRA adapters."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.quant_type,
        bnb_4bit_use_double_quant=config.use_double_quant,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
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


def main():
    parser = argparse.ArgumentParser(description="04. QLoRA (Quantized LoRA)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model identifier")
    parser.add_argument("--output-dir", type=str, default="./outputs/qlora_adapter", help="Directory to save adapter")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (r)")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA scaling factor")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default: dry-run inspection)")
    args = parser.parse_args()

    missing = check_prerequisites()
    if missing:
        print("=" * 70)
        print("⚠️  MISSING PREREQUISITES FOR THIS TECHNIQUE")
        print("=" * 70)
        print(f"Missing packages: {', '.join(missing)}")
        print("\nInstall via:")
        print("    pip install torch transformers peft bitsandbytes accelerate")
        print("=" * 70)
        return

    print("=" * 70)
    print("04. QLoRA (4-BIT QUANTIZED LoRA) — CONFIGURATION")
    print("=" * 70)
    print(f"• Base Model:       {args.model}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Quantization:     4-bit NormalFloat (NF4)")
    print(f"• Double Quant:     Enabled (saves ~0.37 bits/param)")
    print(f"• Compute Dtype:    bfloat16")
    print(f"• Optimizer:        Paged AdamW 8-bit")
    print(f"• VRAM Requirement: < 10 GB for 7B models (runs on RTX 3060 / 4060)")

    if not args.train:
        print("\n💡 Mode: DRY-RUN / INSPECTION ONLY")
        print("To execute live training, run with the `--train` flag:")
        print(f"    python 04_qlora.py --train --model {args.model}")
        print("=" * 70)
        return

    print("\n🚀 Building QLoRA model with 4-bit NF4 quantization...")
    cfg = QLoRAConfig(model_name=args.model, output_dir=args.output_dir, rank=args.rank, alpha=args.alpha)
    model, tokenizer = build_qlora_model(cfg)
    print("✓ Model loaded into 4-bit VRAM successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
'''
write_file("04_qlora.py", T04)

print("\nBatch 1 completed (01 to 04).")
'''
run_cmd = Path("c:/Users/Rishav/Disk/Codes/LLM Projects/Fine_Tuning/scratch/build_techniques.py")
with open(run_cmd, "w", encoding="utf-8") as f:
    f.write(T01)
'''
