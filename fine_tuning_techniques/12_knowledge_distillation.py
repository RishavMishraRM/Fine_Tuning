"""
12. Knowledge Distillation (KD)
================================
Knowledge Distillation (Hinton et al., 2015; Kim & Rush, 2016) transfers the capabilities,
dark knowledge, and probability distribution of an overparameterized Teacher model
(e.g., 70B parameters) into a lightweight Student model (e.g., 1B-3B parameters).

Objective Formulation:
    L_{total} = (1 - alpha) * L_{Task}(y, sigma(z_s)) + alpha * tau^2 * D_{KL}( sigma(z_t / tau) || sigma(z_s / tau) )

Where:
- z_t and z_s are the unnormalized logits of Teacher and Student respectively.
- tau is the distillation Temperature (tau > 1 softens probability distribution over non-target tokens).
- alpha is the weighting factor balancing ground-truth cross-entropy and teacher matching.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


@dataclass
class DistillationConfig:
    temperature: float = 2.0  # Logit softening temperature tau
    alpha: float = 0.5        # Loss balance weight


class DistillationTrainer(Trainer):
    """Custom Trainer implementing Kullback-Leibler divergence logit distillation."""

    def __init__(self, teacher_model=None, distillation_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distillation_config = distillation_config or DistillationConfig()

        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        """Overrides loss computation to blend Student CrossEntropy + KL(Teacher || Student)."""
        # Student forward pass
        student_outputs = model(**inputs)
        student_loss = student_outputs.loss
        student_logits = student_outputs.logits

        # Teacher forward pass (inference mode, no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        tau = self.distillation_config.temperature
        alpha = self.distillation_config.alpha

        # Softened KL Divergence: D_KL(P_Teacher || P_Student)
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / tau, dim=-1),
            F.softmax(teacher_logits / tau, dim=-1),
            reduction="batchmean",
        ) * (tau ** 2)

        total_loss = (1.0 - alpha) * student_loss + alpha * kl_loss

        return (total_loss, student_outputs) if return_outputs else total_loss


def setup_distillation_pipeline(
    teacher_name: str = "mistralai/Mistral-7B-v0.1",
    student_name: str = "Qwen/Qwen2.5-0.5B",
):
    """Sets up Teacher-Student Distillation Trainer."""
    tokenizer = AutoTokenizer.from_pretrained(student_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    student_model = AutoModelForCausalLM.from_pretrained(
        student_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    sample_data = [
        {"text": "Banking query: What is loan pre-closure? Response: Paying loan before tenure ends."},
        {"text": "Banking query: What is UPI mandate? Response: Recurring payment authorization."},
    ]
    dataset = Dataset.from_list(sample_data)

    def tokenize_fn(x):
        return tokenizer(x["text"], truncation=True, max_length=256, padding="max_length")

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])

    training_args = TrainingArguments(
        output_dir="./outputs/distilled_student",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        bf16=True,
        logging_steps=5,
        report_to="none",
    )

    distill_trainer = DistillationTrainer(
        teacher_model=teacher_model,
        distillation_config=DistillationConfig(temperature=2.5, alpha=0.6),
        model=student_model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=default_data_collator,
    )
    return distill_trainer


if __name__ == "__main__":
    print("Knowledge Distillation (KD) Teacher-Student pipeline ready.")
    print("Transfers rich probability logit distribution from heavy Teacher to compact Student.")
