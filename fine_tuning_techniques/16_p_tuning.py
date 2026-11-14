"""
16. P-Tuning (v1 / v2)
=======================
Introduced by Liu et al. (ACL 2022).

Key Insight:
Simple Prompt Tuning treats virtual prompt embeddings as independent parameters,
failing to capture sequential dependencies between tokens and making optimization
sensitive to initialization.

P-Tuning Innovations:
1. Prompt Encoder (P-Tuning v1):
   Passes virtual token indices through an auxiliary neural network (typically a
   bidirectional LSTM followed by a 2-layer MLP) to model sequential correlation:
       h_i = MLP( BiLSTM( e_1, e_2, ..., e_p )_i )
   Once training completes, the LSTM/MLP is discarded, keeping only the generated embeddings.
2. Deep Prompt Tuning (P-Tuning v2):
   Extends virtual prompt tokens across all transformer layers (not just the input layer),
   bridging the gap with Prefix Tuning for challenging language understanding benchmarks.
"""

import torch
from datasets import Dataset
from peft import (
    PromptEncoderConfig,
    PromptEncoderReparameterizationType,
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


def setup_p_tuning_pipeline(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    num_virtual_tokens: int = 20,
    encoder_hidden_size: int = 128,
):
    """Sets up P-Tuning with an LSTM/MLP Prompt Encoder."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    peft_config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=num_virtual_tokens,
        encoder_hidden_size=encoder_hidden_size,
        encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM,
        encoder_num_layers=2,
        encoder_dropout=0.1,
    )

    p_tuned_model = get_peft_model(base_model, peft_config)
    p_tuned_model.print_trainable_parameters()

    sample_data = [
        {"text": "Account update: Customer updated phone number. Category: Profile Modification"},
        {"text": "Account update: International ATM limit increased. Category: Limit Modification"},
    ]
    dataset = Dataset.from_list(sample_data)

    def tokenize_fn(x):
        return tokenizer(x["text"], truncation=True, max_length=128, padding="max_length")

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])

    training_args = TrainingArguments(
        output_dir="./outputs/p_tuning_model",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        num_train_epochs=5,
        bf16=True,
        logging_steps=5,
        report_to="none",
    )

    trainer = Trainer(
        model=p_tuned_model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=default_data_collator,
    )
    return trainer


if __name__ == "__main__":
    print("P-Tuning (with BiLSTM + MLP reparameterization) configured.")
    print("Models sequential token dependencies in virtual prompts for improved convergence.")
