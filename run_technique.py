#!/usr/bin/env python3
"""
================================================================================
🚀 LLM Fine-Tuning Hub — Master Technique Runner
================================================================================
A unified command-line interface to inspect, dry-run, or execute any of the
20 fine-tuning, adaptation, pretraining, distillation, and alignment techniques.

Usage:
    # 1. List all 20 implemented techniques with descriptions & VRAM tiers:
    python run_technique.py --list

    # 2. Dry-run an individual technique (prints architecture, math & VRAM specs):
    python run_technique.py 03
    python run_technique.py lora
    python run_technique.py 04_qlora

    # 3. Execute live training for any technique:
    python run_technique.py 03 --train
    python run_technique.py 04 --train --model Qwen/Qwen2.5-0.5B --epochs 3

    # 4. Check system prerequisites & GPU status:
    python run_technique.py --check-env
================================================================================
"""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TECHNIQUES_DIR = PROJECT_ROOT / "fine_tuning_techniques"

# Registry of all 20 implemented techniques
TECHNIQUES = [
    {
        "id": "01",
        "name": "full_fine_tuning",
        "file": "01_full_fine_tuning.py",
        "category": "Full Tuning",
        "title": "Full Fine-Tuning (FFT)",
        "vram": "Multi-GPU (80GB+)",
        "desc": "Backpropagates gradients across 100% of network parameters. Highest capability shift, high compute cost.",
    },
    {
        "id": "02",
        "name": "peft_overview",
        "file": "02_peft_overview.py",
        "category": "PEFT Analysis",
        "title": "PEFT Parameter Footprint Analysis",
        "vram": "< 12 GB",
        "desc": "Empirical comparison of parameter ratios across LoRA, Prefix Tuning, Prompt Tuning, and P-Tuning.",
    },
    {
        "id": "03",
        "name": "lora",
        "file": "03_lora.py",
        "category": "PEFT",
        "title": "Low-Rank Adaptation (LoRA)",
        "vram": "16 - 24 GB",
        "desc": "Decomposes weight deltas into low-rank matrices W = W_0 + (alpha/r)*BA for zero inference latency.",
    },
    {
        "id": "04",
        "name": "qlora",
        "file": "04_qlora.py",
        "category": "PEFT",
        "title": "Quantized LoRA (QLoRA)",
        "vram": "6 - 12 GB",
        "desc": "4-bit NormalFloat (NF4) quantization + Double Quant + Paged Optimizers for consumer GPU fine-tuning.",
    },
    {
        "id": "05",
        "name": "sft",
        "file": "05_sft.py",
        "category": "Supervised",
        "title": "Supervised Fine-Tuning (SFT)",
        "vram": "16 - 24 GB",
        "desc": "Adapts base foundation LLMs to chat and dialogue using TRL SFTTrainer and conversational packing.",
    },
    {
        "id": "06",
        "name": "instruction_tuning",
        "file": "06_instruction_tuning.py",
        "category": "Supervised",
        "title": "Instruction Fine-Tuning (Masked Loss)",
        "vram": "16 - 24 GB",
        "desc": "Response-only loss masking with DataCollatorForCompletionOnlyLM (label -100 for prompt tokens).",
    },
    {
        "id": "07",
        "name": "cpt",
        "file": "07_cpt.py",
        "category": "Pretraining",
        "title": "Continued Pretraining (CPT)",
        "vram": "24 - 48 GB",
        "desc": "Second-stage pretraining on unstructured text via continuous token concatenation and chunk packing.",
    },
    {
        "id": "08",
        "name": "dapt",
        "file": "08_dapt.py",
        "category": "Pretraining",
        "title": "Domain-Adaptive Pretraining (DAPT)",
        "vram": "24 - 48 GB",
        "desc": "Domain absorption on specialized corpora (finance, legal, biomed) tracking Domain Perplexity (PPL).",
    },
    {
        "id": "09",
        "name": "tapt",
        "file": "09_tapt.py",
        "category": "Pretraining",
        "title": "Task-Adaptive Pretraining (TAPT)",
        "vram": "16 - 24 GB",
        "desc": "Hyper-targeted pretraining on small unlabeled downstream task queries directly prior to SFT.",
    },
    {
        "id": "10",
        "name": "rlhf_ppo",
        "file": "10_rlhf_ppo.py",
        "category": "Alignment",
        "title": "Classical RLHF (SFT + RM + PPO)",
        "vram": "48 - 80 GB",
        "desc": "InstructGPT 3-stage alignment: SFT -> Bradley-Terry Reward Model -> PPO with KL divergence penalty.",
    },
    {
        "id": "11",
        "name": "dpo",
        "file": "11_dpo.py",
        "category": "Alignment",
        "title": "Direct Preference Optimization (DPO)",
        "vram": "16 - 24 GB",
        "desc": "Eliminates reward model: analytically optimizes policy on pairwise (prompt, chosen, rejected) tuples.",
    },
    {
        "id": "12",
        "name": "knowledge_distillation",
        "file": "12_knowledge_distillation.py",
        "category": "Distillation",
        "title": "Knowledge Distillation (KD)",
        "vram": "24 - 80 GB",
        "desc": "Transfers dark knowledge from heavy Teacher LLM to lightweight Student model via softened logits.",
    },
    {
        "id": "13",
        "name": "adapter_tuning",
        "file": "13_adapter_tuning.py",
        "category": "PEFT",
        "title": "Bottleneck Residual Adapters",
        "vram": "< 12 GB",
        "desc": "Houlsby/Pfeiffer series residual adapters (d -> m -> d bottleneck) injected after attention and FFN.",
    },
    {
        "id": "14",
        "name": "prefix_tuning",
        "file": "14_prefix_tuning.py",
        "category": "Soft Prompts",
        "title": "Prefix Tuning",
        "vram": "< 12 GB",
        "desc": "Injects virtual learnable Key/Value attention prefixes (P_K, P_V) across every transformer layer.",
    },
    {
        "id": "15",
        "name": "prompt_tuning",
        "file": "15_prompt_tuning.py",
        "category": "Soft Prompts",
        "title": "Prompt Tuning (Soft Prompts)",
        "vram": "< 8 GB",
        "desc": "Prepends continuous learnable virtual token embeddings exclusively to the input word embedding layer.",
    },
    {
        "id": "16",
        "name": "p_tuning",
        "file": "16_p_tuning.py",
        "category": "Soft Prompts",
        "title": "P-Tuning (v1 / v2)",
        "vram": "< 12 GB",
        "desc": "Uses BiLSTM + MLP prompt encoder to capture sequential dependencies between virtual prompt tokens.",
    },
    {
        "id": "17",
        "name": "orpo",
        "file": "17_orpo.py",
        "category": "Alignment",
        "title": "Odds Ratio Preference Optimization (ORPO)",
        "vram": "16 - 24 GB",
        "desc": "Single-stage SFT + alignment via Odds Ratio penalty. Completely eliminates the frozen reference model.",
    },
    {
        "id": "18",
        "name": "kto",
        "file": "18_kto.py",
        "category": "Alignment",
        "title": "Kahneman-Tversky Optimization (KTO)",
        "vram": "16 - 24 GB",
        "desc": "Aligns LLMs on real unpaired binary telemetry (thumbs up/down) based on behavioral Prospect Theory.",
    },
    {
        "id": "19",
        "name": "ppo",
        "file": "19_ppo.py",
        "category": "Alignment Engine",
        "title": "PPO Mathematical Engine",
        "vram": "24 - 48 GB",
        "desc": "Deep mathematical engine: Clipped Surrogate Objective, Generalized Advantage Estimation (GAE), Critic head.",
    },
    {
        "id": "20",
        "name": "grpo",
        "file": "20_grpo.py",
        "category": "Reasoning & RL",
        "title": "Group Relative Policy Optimization (GRPO)",
        "vram": "24 - 48 GB",
        "desc": "DeepSeek-R1 style RL without a Critic: samples group of G outputs and normalizes relative advantages.",
    },
]


def print_banner():
    print("=" * 80)
    print("🚀 LLM FINE-TUNING HUB — MASTER TECHNIQUE RUNNER")
    print("=" * 80)


def list_techniques():
    print_banner()
    print(f"{'#':<4} {'ID':<4} {'Name':<24} {'Category':<16} {'VRAM Tier':<16} {'File'}")
    print("-" * 80)
    for i, t in enumerate(TECHNIQUES, 1):
        print(f"{i:<4} {t['id']:<4} {t['name']:<24} {t['category']:<16} {t['vram']:<16} {t['file']}")
    print("-" * 80)
    print("\nTo run or inspect any technique:")
    print("  python run_technique.py <id_or_name>          # Architecture & parameter dry-run")
    print("  python run_technique.py <id_or_name> --train  # Execute actual neural network training")
    print("\nExamples:")
    print("  python run_technique.py 03                    # Inspect LoRA")
    print("  python run_technique.py qlora --train         # Train with QLoRA")
    print("  python run_technique.py 11 --train            # Train with DPO")
    print("  python run_technique.py grpo                  # Inspect GRPO (DeepSeek-R1)")
    print("=" * 80)


def check_env():
    print_banner()
    print("🔍 CHECKING RUNTIME ENVIRONMENT & DEPENDENCIES\n")

    # Python Version
    print(f"• Python Version: {sys.version.split()[0]}")

    # Core Ollama
    try:
        import ollama
        print(f"• Ollama Python Library: Installed (v{getattr(ollama, '__version__', 'available')})")
    except ImportError:
        print("• Ollama Python Library: NOT installed (`pip install ollama`)")

    # PyTorch & GPU
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "No CUDA GPU detected"
        print(f"• PyTorch: Installed (v{torch.__version__})")
        print(f"• CUDA Available: {cuda_avail} ({gpu_name})")
        if cuda_avail:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"• GPU Memory: {vram_gb:.2f} GB VRAM")
    except ImportError:
        print("• PyTorch: NOT installed (`pip install torch`)")

    # Training Suite
    packages = {
        "transformers": "HuggingFace Transformers",
        "peft": "PEFT (LoRA, QLoRA, Prefix Tuning)",
        "trl": "TRL (SFT, DPO, PPO, ORPO, GRPO)",
        "datasets": "HuggingFace Datasets",
        "accelerate": "HuggingFace Accelerate",
        "bitsandbytes": "BitsAndBytes (4-bit / 8-bit Quantization)",
    }

    print("\n📦 Training Suite Packages:")
    missing = []
    for pkg, label in packages.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed")
            print(f"  ✓ {pkg:<14s} [{version}] - {label}")
        except ImportError:
            print(f"  ✗ {pkg:<14s} [MISSING]   - {label}")
            missing.append(pkg)

    if missing:
        print("\n💡 To install missing training packages:")
        print("   pip install -r requirements-training.txt")
    else:
        print("\n✨ All training suite dependencies are installed and ready!")
    print("=" * 80)


def resolve_technique(query: str):
    query = query.strip().lower()
    for t in TECHNIQUES:
        if query in (t["id"], t["name"], t["file"], t["file"].replace(".py", "")):
            return t
        if query in t["name"].replace("_", ""):
            return t
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Master runner for 20 LLM fine-tuning techniques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "technique",
        nargs="?",
        help="Technique identifier (number 01-20 or name e.g. 'lora', 'qlora', 'dpo', 'grpo')",
    )
    parser.add_argument("--list", action="store_true", help="List all 20 techniques and exit")
    parser.add_argument("--check-env", action="store_true", help="Check installed dependencies and GPU status")
    parser.add_argument("--train", action="store_true", help="Execute actual training (default is dry-run / config inspection)")
    parser.add_argument("--model", type=str, default=None, help="Override base model identifier")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory for saved weights")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size per device")

    args, extra_args = parser.parse_known_args()

    if args.list:
        list_techniques()
        return

    if args.check_env:
        check_env()
        return

    if not args.technique:
        list_techniques()
        return

    tech = resolve_technique(args.technique)
    if not tech:
        print(f"\n❌ Error: Unknown technique '{args.technique}'")
        print("Run `python run_technique.py --list` to view all available techniques.")
        sys.exit(1)

    target_file = TECHNIQUES_DIR / tech["file"]
    if not target_file.exists():
        print(f"\n❌ Error: File not found: {target_file}")
        sys.exit(1)

    # Build command line to execute the target file standalone
    cmd = [sys.executable, str(target_file)]
    if args.train:
        cmd.append("--train")
    if args.model:
        cmd.extend(["--model", args.model])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    cmd.extend(extra_args)

    print_banner()
    print(f"Executing: {tech['title']}")
    print(f"Script:    {target_file.relative_to(PROJECT_ROOT)}")
    print(f"Mode:      {'LIVE TRAINING' if args.train else 'DRY-RUN / CONFIG INSPECTION'}")
    print("=" * 80 + "\n")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
