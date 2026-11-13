"""
Evaluate the banking-classifier model on intent classification.

Compares accuracy of the base mistral model vs the custom classifier.

Usage:
    python src/classify.py
    python src/classify.py --compare
"""

import argparse
import csv
import sys
from pathlib import Path

import ollama


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.test_cases import test_cases


DEFAULT_MODEL = "banking-classifier"
BASE_MODEL = "mistral:latest"

BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Classify the banking intent. "
    "Return only the category name: Payment, Refund, Loan, or Credit Card."
)

ALLOWED_CATEGORIES = ["Credit Card", "Payment", "Refund", "Loan"]


def clean_category(raw_output):
    """Normalize model output to exact allowed category."""
    cleaned = raw_output.strip().rstrip(".").strip()
    
    # Direct match first
    for cat in ALLOWED_CATEGORIES:
        if cleaned.lower() == cat.lower():
            return cat
            
    # Starts with category
    for cat in ALLOWED_CATEGORIES:
        if cleaned.lower().startswith(cat.lower()):
            return cat

    # Contained in response
    for cat in ALLOWED_CATEGORIES:
        if cat.lower() in cleaned.lower():
            return cat
            
    return cleaned


def classify(message, model=DEFAULT_MODEL, system_prompt=None):
    """Classify a banking message."""

    messages = [{"role": "user", "content": message}]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    response = ollama.chat(
        model=model,
        messages=messages,
        options={
            "temperature": 0,
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": 10,
        },
    )

    raw = response["message"]["content"].strip()
    return clean_category(raw)


def evaluate_classifier(model, cases, system_prompt=None):
    """Run classification on all test cases."""

    correct = 0
    results = []

    for i, test in enumerate(cases, 1):
        actual = classify(test["message"], model, system_prompt)
        passed = actual.lower() == test["expected"].lower()

        if passed:
            correct += 1

        results.append({
            "message": test["message"],
            "expected": test["expected"],
            "actual": actual,
            "result": "PASS" if passed else "FAIL",
        })

        status = "PASS" if passed else "FAIL"
        print(f"  {i:2d}. [{status}] Expected: {test['expected']:12s} | Got: {actual}", flush=True)

    accuracy = (correct / len(cases)) * 100 if cases else 0

    return results, accuracy, correct


def main():
    parser = argparse.ArgumentParser(description="Banking intent classification")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--compare", action="store_true", help="Compare base vs custom")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    if args.compare:
        print("=" * 60, flush=True)
        print("CLASSIFIER COMPARISON", flush=True)
        print("=" * 60, flush=True)

        print(f"\n--- BASE: {BASE_MODEL} ---\n", flush=True)
        base_results, base_acc, base_correct = evaluate_classifier(
            BASE_MODEL, test_cases, BASE_SYSTEM_PROMPT,
        )

        print(f"\n--- CUSTOM: {DEFAULT_MODEL} ---\n", flush=True)
        custom_results, custom_acc, custom_correct = evaluate_classifier(
            DEFAULT_MODEL, test_cases,
        )

        print("\n" + "=" * 60, flush=True)
        print("COMPARISON RESULTS", flush=True)
        print("=" * 60, flush=True)
        print(f"{'Model':<25s} {'Accuracy':>10s} {'Correct':>10s} {'Total':>8s}", flush=True)
        print("-" * 60, flush=True)
        print(f"{BASE_MODEL:<25s} {base_acc:>9.2f}% {base_correct:>10d} {len(test_cases):>8d}", flush=True)
        print(f"{DEFAULT_MODEL:<25s} {custom_acc:>9.2f}% {custom_correct:>10d} {len(test_cases):>8d}", flush=True)
        print(f"{'Improvement':<25s} {custom_acc - base_acc:>+9.2f}%", flush=True)
        print("=" * 60, flush=True)

        # Save comparison
        with open(output_dir / "classifier_comparison.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "message", "expected", "base_actual", "custom_actual",
                "base_result", "custom_result",
            ])
            writer.writeheader()
            for b, c in zip(base_results, custom_results):
                writer.writerow({
                    "message": b["message"],
                    "expected": b["expected"],
                    "base_actual": b["actual"],
                    "custom_actual": c["actual"],
                    "base_result": b["result"],
                    "custom_result": c["result"],
                })

        print(f"\nSaved to: {output_dir / 'classifier_comparison.csv'}", flush=True)

    else:
        print("=" * 60, flush=True)
        print(f"EVALUATING CLASSIFIER: {args.model}", flush=True)
        print(f"Test cases: {len(test_cases)}", flush=True)
        print("=" * 60 + "\n", flush=True)

        results, accuracy, correct = evaluate_classifier(args.model, test_cases)

        print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{len(test_cases)})", flush=True)

        # Save results
        with open(output_dir / "classification_report.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["message", "expected", "actual", "result"])
            writer.writeheader()
            writer.writerows(results)

        print(f"Saved to: {output_dir / 'classification_report.csv'}", flush=True)


if __name__ == "__main__":
    main()
