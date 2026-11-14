"""
Datasets Package
================
Provides domain benchmark scenarios and classification test cases for evaluating
fine-tuned models and verifying compliance guardrails.

Available Datasets:
- `evaluation_cases`: 30 multi-point domain advisory scenarios with expected guidance
  and explicitly forbidden assertions for generative evaluation.
- `test_cases`: 30 multi-intent customer queries with expected ground-truth category tags
  for intent classification benchmarks.

How to Add Your Own Domain Dataset:
1. Define a list of dictionaries adhering to the schema in `evaluation_cases.py`:
   - `id`: Unique integer identifier
   - `category`: Domain vertical / topic
   - `question`: Customer / user prompt
   - `expected_points`: List of key advisory points the model should mention
   - `forbidden_claims`: Disallowed phrases or claims the model must avoid
2. Export it from this module and reference it in `src/evaluate.py`.
"""

from datasets.evaluation_cases import evaluation_cases
from datasets.test_cases import test_cases

__all__ = ["evaluation_cases", "test_cases"]
