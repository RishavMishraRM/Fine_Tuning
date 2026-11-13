"""
Create custom Ollama models from Modelfiles.

This script builds two custom models:
1. banking-assistant  - Customer support response model
2. banking-classifier - Intent classification model

Usage:
    python src/create_models.py
"""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    {
        "name": "banking-assistant",
        "modelfile": PROJECT_ROOT / "Modelfile",
        "description": "Banking customer support assistant",
    },
    {
        "name": "banking-classifier",
        "modelfile": PROJECT_ROOT / "Modelfile.classifier",
        "description": "Banking intent classifier",
    },
]


def create_model(name, modelfile):
    """Create an Ollama model from a Modelfile."""

    print(f"\nCreating model: {name}")
    print(f"Modelfile: {modelfile}")

    if not modelfile.exists():
        print(f"ERROR: Modelfile not found: {modelfile}")
        return False

    result = subprocess.run(
        ["ollama", "create", name, "-f", str(modelfile)],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(f"SUCCESS: Model '{name}' created")
        if result.stdout.strip():
            print(result.stdout.strip())
        return True
    else:
        print(f"FAILED: {result.stderr.strip()}")
        return False


def verify_model(name):
    """Verify a model exists in Ollama."""

    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        return name in result.stdout

    return False


def main():
    print("=" * 60)
    print("CREATING CUSTOM OLLAMA MODELS")
    print("=" * 60)

    # Check Ollama is available
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
        )
        print(f"Ollama version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("ERROR: Ollama is not installed or not in PATH")
        sys.exit(1)

    results = []

    for model in MODELS:
        success = create_model(model["name"], model["modelfile"])
        results.append((model["name"], model["description"], success))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, description, success in results:
        status = "CREATED" if success else "FAILED"
        print(f"  {name:25s} [{status}] - {description}")

    print("\nTo test a model:")
    print("  ollama run banking-assistant")
    print("  ollama run banking-classifier")

    print("\nTo use in Python:")
    print('  import ollama')
    print('  response = ollama.chat(model="banking-assistant", messages=[...])')


if __name__ == "__main__":
    main()

