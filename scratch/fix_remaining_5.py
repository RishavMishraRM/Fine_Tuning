from pathlib import Path
import re

TECH_DIR = Path("c:/Users/Rishav/Disk/Codes/LLM Projects/Fine_Tuning/fine_tuning_techniques")

# 1. Fix 01_full_fine_tuning.py
f01 = TECH_DIR / "01_full_fine_tuning.py"
c01 = f01.read_text(encoding="utf-8")
c01 = c01.replace("""def prepare_sample_dataset(tokenizer, max_length: int = 512):
    \"\"\"Generates a tokenized causal LM dataset for general domain adaptation.\"\"\"
try:
        from datasets import Dataset
except ImportError:
    pass""", """def prepare_sample_dataset(tokenizer, max_length: int = 512):
    \"\"\"Generates a tokenized causal LM dataset for general domain adaptation.\"\"\"
    try:
        from datasets import Dataset
    except ImportError:
        return None""")
f01.write_text(c01, encoding="utf-8")
print("Fixed 01")

# 2. Fix 02_peft_overview.py
f02 = TECH_DIR / "02_peft_overview.py"
c02 = f02.read_text(encoding="utf-8")
c02 = c02.replace('    print("\nPEFT Configuration Matrix:")', '    print("\\nPEFT Configuration Matrix:")')
c02 = c02.replace('    print("\nTheoretical Comparison Table', '    print("\\nTheoretical Comparison Table')
f02.write_text(c02, encoding="utf-8")
print("Fixed 02")

# 3. Fix 03_lora.py
f03 = TECH_DIR / "03_lora.py"
c03 = f03.read_text(encoding="utf-8")
c03 = c03.replace("""def build_sample_dataset(tokenizer, max_length: int = 512):
try:
        from datasets import Dataset
except ImportError:
    pass""", """def build_sample_dataset(tokenizer, max_length: int = 512):
    try:
        from datasets import Dataset
    except ImportError:
        return None""")
f03.write_text(c03, encoding="utf-8")
print("Fixed 03")

# 4. Fix 04_qlora.py
f04 = TECH_DIR / "04_qlora.py"
c04 = f04.read_text(encoding="utf-8")
c04 = re.sub(
    r"def build_qlora_model\(config: QLoRAConfig\):[\s\S]*?return qlora_model, tokenizer",
    """def build_qlora_model(config: QLoRAConfig):
    \"\"\"Initializes 4-bit quantized base model and attaches trainable FP32/BF16 LoRA adapters.\"\"\"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

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

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

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
    return qlora_model, tokenizer""",
    c04
)
f04.write_text(c04, encoding="utf-8")
print("Fixed 04")

# 5. Fix 20_grpo.py
f20 = TECH_DIR / "20_grpo.py"
c20 = f20.read_text(encoding="utf-8")
c20 = c20.replace("""    try:
try:
            from trl import GRPOConfig, GRPOTrainer
except ImportError:
    pass""", """    try:
        from trl import GRPOConfig, GRPOTrainer""")
f20.write_text(c20, encoding="utf-8")
print("Fixed 20")
