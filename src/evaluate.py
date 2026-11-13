"""
Evaluate the custom banking-assistant model against evaluation cases.

Compares the base mistral model with the custom banking-assistant model
to measure the improvement from Modelfile-based fine-tuning.

Usage:
    python src/evaluate.py
    python src/evaluate.py --model banking-assistant
    python src/evaluate.py --compare
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import ollama


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.evaluation_cases import evaluation_cases


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = "banking-assistant"
BASE_MODEL = "mistral:latest"
NUM_PREDICT = 700
TEMPERATURE = 0

SYSTEM_PROMPT_BASE = """You are a helpful assistant."""

# Stop words for coverage calculation
STOP_WORDS = {
    "about", "after", "again", "also", "because", "before", "being",
    "check", "could", "does", "from", "have", "into", "more", "must",
    "only", "should", "that", "their", "there", "these", "this",
    "through", "under", "using", "when", "where", "which", "while",
    "with", "your",
}


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.lower()).strip()


def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text)


def is_negated(sentence, match_start):
    preceding = sentence[:match_start]
    clauses = re.split(r"[,;.!?]|\b(?:but|however|although|yet)\b", preceding, flags=re.I)
    clause = clauses[-1].strip().lower() if clauses else ""
    return bool(re.search(
        r"\b(?:not|never|no|cannot|can't|shouldn't|should not|do not|don't|"
        r"will not|won't|must not|mustn't|avoid|refrain from)\b",
        clause,
    ))


# ============================================================
# COVERAGE CALCULATION
# ============================================================

def calculate_coverage(response, expected_points):
    """Keyword-based coverage check."""

    response_words = set(re.findall(r"\b[a-z0-9]+\b", normalize_text(response)))

    matched = []
    missing = []

    for point in expected_points:
        keywords = {
            word for word in re.findall(r"\b[a-z0-9]+\b", normalize_text(point))
            if len(word) > 4 and word not in STOP_WORDS
        }

        hits = keywords.intersection(response_words)
        required = max(1, math.ceil(len(keywords) * 0.30)) if keywords else 1
        passed = bool(keywords) and len(hits) >= required

        (matched if passed else missing).append(point)

    total = len(expected_points)
    coverage = round(len(matched) / total * 100, 2) if total else 0

    return {
        "coverage": coverage,
        "matched_points": matched,
        "missing_points": missing,
    }


# ============================================================
# RISK DETECTION
# ============================================================

RISK_PATTERNS = [
    ("CREDENTIAL_REQUEST", "critical", [
        r"\b(?:share|provide|send|tell me|enter)\s+(?:your\s+)?(?:otp|cvv|pin|password)\b",
    ]),
    ("GUARANTEED_OUTCOME", "high", [
        r"\bguaranteed\b",
        r"\bdefinitely\s+(?:be\s+)?(?:refunded|reversed|approved)\b",
        r"\bwill\s+(?:be\s+)?(?:refunded|reversed|approved)\b",
    ]),
    ("SPECIFIC_TIMELINE", "medium", [
        r"\b(?:within|in|after)\s+\d+(?:\s*[-–]\s*\d+)?\s*(?:business\s+)?(?:days?|hours?)\b",
    ]),
]


def detect_risks(response):
    text = normalize_text(response)
    flags = []

    for sentence in split_sentences(text):
        for rule_id, severity, patterns in RISK_PATTERNS:
            for pattern in patterns:
                for match in re.finditer(pattern, sentence):
                    if not is_negated(sentence, match.start()):
                        flags.append({
                            "rule_id": rule_id,
                            "severity": severity,
                            "matched_text": match.group(),
                        })

    return flags


# ============================================================
# MODEL INTERACTION
# ============================================================

def generate_response(question, model=DEFAULT_MODEL, system_prompt=None):
    """Generate a response from an Ollama model."""

    messages = [{"role": "user", "content": question}]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    response = ollama.chat(
        model=model,
        messages=messages,
        options={
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
        },
    )

    message = response.get("message", {})

    return {
        "answer": (message.get("content") or "").strip(),
        "done_reason": response.get("done_reason", ""),
        "eval_count": response.get("eval_count"),
    }


# ============================================================
# SINGLE MODEL EVALUATION
# ============================================================

def evaluate_model(model, cases, system_prompt=None):
    """Evaluate a model against all cases."""

    results = []

    for case in cases:
        print(f"  Case {case['id']:2d} | {case['category']:20s} | ", end="")

        try:
            gen = generate_response(case["question"], model, system_prompt)
            answer = gen["answer"]

            if not answer:
                raise ValueError("Empty response")

            coverage = calculate_coverage(answer, case["expected_points"])
            risks = detect_risks(answer)

            # Check forbidden claims
            response_lower = normalize_text(answer)
            forbidden_found = [
                claim for claim in case.get("forbidden_claims", [])
                if normalize_text(claim) and normalize_text(claim) in response_lower
            ]

            results.append({
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "response": answer,
                "coverage": coverage["coverage"],
                "matched_count": len(coverage["matched_points"]),
                "missing_count": len(coverage["missing_points"]),
                "missing_points": json.dumps(coverage["missing_points"], ensure_ascii=False),
                "risk_count": len(risks),
                "forbidden_found": len(forbidden_found),
                "status": "SUCCESS",
            })

            risk_marker = f" [{len(risks)} risks]" if risks else ""
            forbidden_marker = f" [FORBIDDEN!]" if forbidden_found else ""
            print(f"Coverage: {coverage['coverage']:6.2f}%{risk_marker}{forbidden_marker}")

        except Exception as error:
            results.append({
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "response": "",
                "coverage": 0,
                "matched_count": 0,
                "missing_count": len(case["expected_points"]),
                "missing_points": "[]",
                "risk_count": 0,
                "forbidden_found": 0,
                "status": f"FAILED: {error}",
            })
            print(f"FAILED: {error}")

    return results


# ============================================================
# COMPARISON MODE
# ============================================================

def run_comparison(cases):
    """Compare base model vs custom model."""

    print("\n" + "=" * 70)
    print("MODEL COMPARISON: Base (mistral) vs Custom (banking-assistant)")
    print("=" * 70)

    # Evaluate base model
    print(f"\n--- Evaluating BASE model: {BASE_MODEL} ---\n")
    base_results = evaluate_model(BASE_MODEL, cases, system_prompt=SYSTEM_PROMPT_BASE)

    # Evaluate custom model
    print(f"\n--- Evaluating CUSTOM model: {DEFAULT_MODEL} ---\n")
    custom_results = evaluate_model(DEFAULT_MODEL, cases)

    # Save comparison CSV
    output_file = PROJECT_ROOT / "results" / "model_comparison.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "category", "question",
            "base_coverage", "custom_coverage", "improvement",
            "base_risks", "custom_risks",
            "base_forbidden", "custom_forbidden",
        ])
        writer.writeheader()

        for base, custom in zip(base_results, custom_results):
            writer.writerow({
                "id": base["id"],
                "category": base["category"],
                "question": base["question"],
                "base_coverage": base["coverage"],
                "custom_coverage": custom["coverage"],
                "improvement": custom["coverage"] - base["coverage"],
                "base_risks": base["risk_count"],
                "custom_risks": custom["risk_count"],
                "base_forbidden": base["forbidden_found"],
                "custom_forbidden": custom["forbidden_found"],
            })

    # Print summary
    base_success = [r for r in base_results if r["status"] == "SUCCESS"]
    custom_success = [r for r in custom_results if r["status"] == "SUCCESS"]

    base_avg = sum(r["coverage"] for r in base_success) / len(base_success) if base_success else 0
    custom_avg = sum(r["coverage"] for r in custom_success) / len(custom_success) if custom_success else 0

    base_risks = sum(r["risk_count"] for r in base_success)
    custom_risks = sum(r["risk_count"] for r in custom_success)

    base_forbidden = sum(r["forbidden_found"] for r in base_success)
    custom_forbidden = sum(r["forbidden_found"] for r in custom_success)

    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Metric':<30s} {'Base (mistral)':>15s} {'Custom':>15s} {'Delta':>10s}")
    print("-" * 70)
    print(f"{'Avg Coverage':<30s} {base_avg:>14.2f}% {custom_avg:>14.2f}% {custom_avg - base_avg:>+9.2f}%")
    print(f"{'Total Risk Flags':<30s} {base_risks:>15d} {custom_risks:>15d} {custom_risks - base_risks:>+10d}")
    print(f"{'Forbidden Claims Found':<30s} {base_forbidden:>15d} {custom_forbidden:>15d} {custom_forbidden - base_forbidden:>+10d}")
    print(f"{'Successful Evaluations':<30s} {len(base_success):>15d} {len(custom_success):>15d}")
    print(f"\nComparison saved to: {output_file}")
    print("=" * 70)


# ============================================================
# SINGLE MODEL MODE
# ============================================================

def run_single(model, cases):
    """Evaluate a single model."""

    print("\n" + "=" * 60)
    print(f"EVALUATING MODEL: {model}")
    print(f"Total test cases: {len(cases)}")
    print("=" * 60 + "\n")

    results = evaluate_model(model, cases)

    # Save report
    output_file = PROJECT_ROOT / "results" / "evaluation_report.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys()) if results else []
    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    successful = [r for r in results if r["status"] == "SUCCESS"]
    avg_coverage = sum(r["coverage"] for r in successful) / len(successful) if successful else 0
    total_risks = sum(r["risk_count"] for r in successful)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Successful: {len(successful)}/{len(cases)}")
    print(f"Average coverage: {avg_coverage:.2f}%")
    print(f"Total risk flags: {total_risks}")
    print(f"Report saved to: {output_file}")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate banking LLM models")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to evaluate")
    parser.add_argument("--compare", action="store_true", help="Compare base vs custom model")
    args = parser.parse_args()

    if args.compare:
        run_comparison(evaluation_cases)
    else:
        run_single(args.model, evaluation_cases)


if __name__ == "__main__":
    main()

