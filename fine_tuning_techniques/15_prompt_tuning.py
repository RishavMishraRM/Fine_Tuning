"""
15. Prompt Tuning (Soft Prompting)
===================================
Introduced by Lester, Al-Rfou, and Constant (EMNLP 2021).

Concept:
- Unlike Prefix Tuning (which injects parameters into all transformer layers' K/V projections),
  Prompt Tuning injects continuous learnable virtual tokens ONLY into the input word embedding space.
- Representation:
      X_{embedded} = [ P_{soft} ; E(x_1), E(x_2), ..., E(x_T) ]
      where P_{soft} in R^{p x d} is a matrix of p continuous virtual prompt tokens.
- All subsequent transformer layers remain entirely untouched and frozen.
- Remarkable Property: Lester et al. demonstrated that as model scale exceeds ~10 Billion
  parameters, simple Prompt Tuning matches the task performance of Full Fine-Tuning while
  training less than 0.01% of parameters.
"""

import torch
from datasets import Dataset
from peft import (
    PromptTuningConfig,
    PromptTuningInit,
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


def setup_prompt_tuning_pipeline(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    num_virtual_tokens: int = 20,
    init_text: str = "Classify the financial customer inquiry into standard categories:",
):
    """Sets up Soft Prompt Tuning prepended to the input embedding matrix."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Initialize prompt tokens with embeddings of a natural language string for faster convergence
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=num_virtual_tokens,
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text=init_text,
        tokenizer_name_or_path=model_name,
    )

    prompt_tuned_model = get_peft_model(base_model, peft_config)
    prompt_tuned_model.print_trainable_parameters()

    sample_data = [
        {"text": "Inquiry: Card payment failed at checkout. Intent: Payment"},
        {"text": "Inquiry: Requesting personal loan statement. Intent: Loan"},
    ]
    dataset = Dataset.from_list(sample_data)

    def tokenize_fn(x):
        return tokenizer(x["text"], truncation=True, max_length=128, padding="max_length")

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])

    training_args = TrainingArguments(
        output_dir="./outputs/prompt_tuning_model",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=3e-2,  # Soft prompts typically require significantly higher LR (e.g., 1e-2 to 5e-2)
        num_train_epochs=5,
        bf16=True,
        logging_steps=5,
        report_to="none",
    )

    trainer = Trainer(
        model=prompt_tuned_model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=default_data_collator,
    )
    return trainer


if __name__ == "__main__":
    print("Prompt Tuning (Soft Prompts) module ready.")
    print("Trains a continuous virtual token tensor P in R^{p x d} prepended only to input embeddings.")
