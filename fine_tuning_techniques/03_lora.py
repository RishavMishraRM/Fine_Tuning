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

from dataclasses import dataclass
from typing import List, Optional
import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


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


def build_lora_model(config: LoRAExperimentConfig):
    """Loads base model and attaches LoRA low-rank adapter matrices."""
    targets = config.target_modules or get_default_lora_targets()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
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


if __name__ == "__main__":
    print("LoRA (Low-Rank Adaptation) module ready.")
    cfg = LoRAExperimentConfig()
    print(f"Default config: Rank={cfg.rank}, Alpha={cfg.alpha}, Targets={get_default_lora_targets()}")
