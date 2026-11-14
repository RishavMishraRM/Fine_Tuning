"""
18. KTO (Kahneman-Tversky Optimization)
========================================
Introduced by Ethayarajh et al. (2024).

The Practical Challenge of DPO/RLHF:
DPO and RLHF require paired preference data (prompt x with both chosen y_w AND rejected y_l).
In production systems, real user telemetry arrives as UNPAIRED binary feedback:
a user simply gives a thumbs-up (helpful) or thumbs-down (unhelpful) to a single completion.

Theoretical Foundation:
Grounds model alignment in Daniel Kahneman & Amos Tversky's Prospect Theory (1979),
which models human decision-making: humans are more sensitive to losses than gains
(loss aversion).

Utility Function:
    U(x, y) = lambda_D * ( r(x, y) - z_ref )
    where r(x, y) = beta * log( pi_theta(y|x) / pi_ref(y|x) )
    Loss aversion parameter penalizes undesirable completions more aggressively than
    rewarding desirable ones.

Dataset Structure:
Each example has: prompt, completion, and label (bool: True for desirable, False for undesirable).
No paired comparisons required!
"""

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOConfig, KTOTrainer


def build_kto_unpaired_dataset() -> Dataset:
    """Unpaired binary feedback data (thumbs-up / thumbs-down telemetry)."""
    telemetry_samples = [
        {
            "prompt": "How can I block my lost debit card?",
            "completion": "Open your mobile banking app, navigate to Cards > Manage, and tap Freeze Card immediately.",
            "label": True,  # Thumbs up / Desirable response
        },
        {
            "prompt": "How can I block my lost debit card?",
            "completion": "Lost cards are not dangerous. Just wait a few weeks to see if it turns up in lost and found.",
            "label": False,  # Thumbs down / Undesirable response
        },
        {
            "prompt": "What is an OTP used for?",
            "completion": "An OTP is a temporary security code to authenticate transactions. Never share it with anyone.",
            "label": True,
        },
    ]
    return Dataset.from_list(telemetry_samples)


def setup_kto_trainer(model_name: str = "mistralai/Mistral-7B-v0.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    dataset = build_kto_unpaired_dataset()

    kto_args = KTOConfig(
        output_dir="./outputs/kto_model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        beta=0.1,  # Temperature of the implicit reward
        desirable_weight=1.0,  # Weight on positive feedback
        undesirable_weight=1.33,  # Loss aversion weight (penalizes negatives more heavily)
        max_prompt_length=256,
        max_length=512,
        num_train_epochs=3,
        bf16=True,
        logging_steps=5,
        report_to="none",
    )

    trainer = KTOTrainer(
        model=model,
        ref_model=None,  # Automatically managed frozen reference copy
        args=kto_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    return trainer


if __name__ == "__main__":
    print("Kahneman-Tversky Optimization (KTO) pipeline configured.")
    print("Directly aligns model on unpaired thumbs-up / thumbs-down binary user telemetry.")
