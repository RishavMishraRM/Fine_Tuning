"""
Fine-Tuning Techniques Module
==============================
A reference implementation of 20 modern LLM fine-tuning, adaptation,
pre-training, distillation, and preference alignment techniques.

Techniques included:
01. Full Fine-Tuning (FFT)
02. Parameter-Efficient Fine-Tuning (PEFT Overview)
03. LoRA (Low-Rank Adaptation)
04. QLoRA (Quantized Low-Rank Adaptation)
05. Supervised Fine-Tuning (SFT)
06. Instruction Fine-Tuning
07. Continued Pretraining (CPT)
08. Domain-Adaptive Pretraining (DAPT)
09. Task-Adaptive Pretraining (TAPT)
10. Reinforcement Learning from Human Feedback (RLHF)
11. Direct Preference Optimization (DPO)
12. Knowledge Distillation
13. Adapter-Based Fine-Tuning
14. Prefix Tuning
15. Prompt Tuning
16. P-Tuning (v1 / v2)
17. ORPO (Odds Ratio Preference Optimization)
18. KTO (Kahneman-Tversky Optimization)
19. PPO (Proximal Policy Optimization)
20. GRPO (Group Relative Policy Optimization)
"""

TECHNIQUES = [
    "01_full_fine_tuning",
    "02_peft_overview",
    "03_lora",
    "04_qlora",
    "05_sft",
    "06_instruction_tuning",
    "07_cpt",
    "08_dapt",
    "09_tapt",
    "10_rlhf_ppo",
    "11_dpo",
    "12_knowledge_distillation",
    "13_adapter_tuning",
    "14_prefix_tuning",
    "15_prompt_tuning",
    "16_p_tuning",
    "17_orpo",
    "18_kto",
    "19_ppo",
    "20_grpo",
]
