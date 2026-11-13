# Banking LLM Fine-Tuning via Ollama

A project that creates custom-tuned banking LLM models using Ollama's Modelfile system. Two models are built: a **customer support assistant** and an **intent classifier**, both based on `mistral:latest` and customized with domain-specific system prompts, behavioral constraints, and generation parameters.

---

## What was done and why

### Problem

The original project attempted to fine-tune an LLM for banking tasks but got stuck building an over-engineered evaluation framework (~2500+ lines across 15+ files) without ever performing any actual fine-tuning. It included:

- A 1068-line LLM-as-a-Judge pipeline with structural validation, evidence verification, and score consistency checking
- A local evaluator with 5 sub-modules (runner, coverage, criteria, safety, semantic_assist) totaling 700+ lines
- Experiment scripts for prompt comparison, parameter sweeps, manual review generation, training candidate generation, and judge calibration
- Multiple backup directories, `__pycache__` folders, and 10+ CSV result files
- All of this ran slowly because it made dozens of sequential Ollama calls per evaluation

None of this ever resulted in a fine-tuned model.

### Solution

The project was restructured to actually create custom models using **Ollama's Modelfile approach**, which is the correct way to customize models through Ollama. Here's what changed:

| What | Why |
|---|---|
| Created `Modelfile` | Defines the banking-assistant custom model with a domain-specific system prompt, safety rules, and tuned parameters |
| Created `Modelfile.classifier` | Defines a banking intent classifier with strict classification rules and minimal token output |
| Created `src/create_models.py` | One-command script to build both custom models via `ollama create` |
| Created `src/evaluate.py` | Clean evaluation script with `--compare` flag to measure base vs custom model improvement |
| Created `src/classify.py` | Intent classification evaluator with `--compare` flag for base vs custom comparison |
| Simplified `src/main.py` | Interactive chat with the custom model |
| Kept `datasets/` | The evaluation cases and test cases are good data — they were preserved as-is |
| **Removed** `evaluations/` (7 files) | Over-engineered evaluation framework that was never needed for fine-tuning |
| **Removed** `experiments/` (5 files) | Prompt comparison and parameter sweep experiments — useful for research but blocked actual fine-tuning progress |
| **Removed** `evaluations/local_evaluator/` (6 files) | Semantic evaluator with checkpoint support — unnecessary complexity |
| **Removed** `prompts/` (4 files) | Empty prompt template files that were never populated |
| **Removed** `backup_before_local_evaluator_*` | Old backup directory |
| **Removed** old CSV results (10+ files) | Stale outputs from the old evaluation pipeline |

### How Ollama "fine-tuning" works

Ollama does not support traditional weight-based fine-tuning (LoRA/QLoRA) directly. Instead, it provides **Modelfile-based customization**, which:

1. Takes a base model (e.g., `mistral:latest`)
2. Bakes in a system prompt that shapes behavior permanently
3. Sets default generation parameters (temperature, top_p, etc.)
4. Creates a new named model that can be used like any other Ollama model

This is the practical approach for your hardware (RTX 3050, 4 GB VRAM), which cannot support traditional LoRA/QLoRA training of 7B models locally.

> **Note:** If you later want true weight-based fine-tuning, you would need to:
> 1. Train with Unsloth/PEFT on a cloud GPU (e.g., Google Colab)
> 2. Export the trained model as a `.gguf` file
> 3. Replace `FROM mistral:latest` with `FROM ./your_trained_model.gguf` in the Modelfile
> 4. Run `ollama create` again

---

## Project structure

```
Fine_Tuning/
├── Modelfile                  # Custom model definition (banking assistant)
├── Modelfile.classifier       # Custom model definition (intent classifier)
├── README.md
├── requirements.txt
├── datasets/
│   ├── evaluation_cases.py    # 30 banking scenarios with expected points and forbidden claims
│   └── test_cases.py          # 30 intent classification test cases
├── results/                   # Generated CSV reports (created by running scripts)
│   └── .gitkeep
└── src/
    ├── create_models.py       # Build custom Ollama models from Modelfiles
    ├── evaluate.py            # Evaluate and compare banking assistant models
    ├── classify.py            # Evaluate and compare intent classifier models
    └── main.py                # Interactive chat with the banking assistant
```

---

## Prerequisites

1. **Windows 10/11** (or another OS supported by Ollama)
2. **Python 3.10+** installed
3. **Ollama** installed and running
4. **Mistral model** pulled locally

Check your setup:

```powershell
python --version
ollama --version
ollama list
```

---

## Installation

```powershell
cd "C:\Users\Rishav\Disk\Codes\LLM Projects\Fine_Tuning"

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Pull the base model (if not already installed)
ollama pull mistral
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

---

## Step 1: Create the custom models

This builds two custom models from the Modelfiles:

```powershell
python src/create_models.py
```

This runs `ollama create` for each model. You should see:

```
Creating model: banking-assistant
SUCCESS: Model 'banking-assistant' created

Creating model: banking-classifier
SUCCESS: Model 'banking-classifier' created
```

Verify the models exist:

```powershell
ollama list
```

You should see `banking-assistant` and `banking-classifier` in the list alongside `mistral:latest`.

---

## Step 2: Test the models interactively

### Chat with the banking assistant

```powershell
ollama run banking-assistant
```

Try questions like:
- "I paid my credit card bill but it still shows overdue"
- "Someone used my credit card without permission"
- "What are the foreclosure charges on my personal loan?"

Type `/bye` to exit.

### Test the classifier

```powershell
ollama run banking-classifier
```

Type a banking message and it should return just the category name.

---

## Step 3: Evaluate the models

### Evaluate the banking assistant

Run on the custom model (30 banking scenarios):

```powershell
python src/evaluate.py
```

Compare base `mistral` vs custom `banking-assistant`:

```powershell
python src/evaluate.py --compare
```

This outputs:
- `results/evaluation_report.csv` — detailed per-case results
- `results/model_comparison.csv` — side-by-side comparison (when using `--compare`)

### Evaluate the intent classifier

Run on the custom classifier (30 test cases):

```powershell
python src/classify.py
```

Compare base vs custom:

```powershell
python src/classify.py --compare
```

This outputs:
- `results/classification_report.csv` — per-case classification results
- `results/classifier_comparison.csv` — side-by-side comparison (when using `--compare`)

### Interactive chat via Python

```powershell
python src/main.py
```

Or use the base model for comparison:

```powershell
python src/main.py --model mistral:latest
```

---

## Evaluation metrics

### Coverage (Banking Assistant)

Keyword-based check measuring how many expected response points the model addressed. Each evaluation case has 4 expected points (e.g., "Acknowledge the customer's payment", "Suggest verifying the transaction"). A point is considered covered when at least 30% of its significant keywords appear in the response.

**Limitation:** This is a lexical heuristic, not semantic understanding. A high score doesn't guarantee the response is correct.

### Risk flags (Banking Assistant)

Automated detection of potentially problematic patterns in responses:

| Severity | Type | Example |
|---|---|---|
| Critical | Credential request | "Share your OTP" |
| High | Guaranteed outcome | "Will definitely be refunded" |
| Medium | Specific timeline | "Within 3-5 business days" |

A flag requires review — it is not proof of a violation.

### Forbidden claims (Banking Assistant)

Each evaluation case specifies claims the model must not make (e.g., "The payment will definitely reflect tomorrow"). The evaluator checks for exact matches.

### Accuracy (Classifier)

Percentage of test cases where the model returned the correct category.

---

## Evaluation dataset

### Banking assistant scenarios (30 cases)

| Category | Count | Examples |
|---|---|---|
| Payment | 5 | Bill overdue, duplicate payment, declined payment |
| Refund and Disputes | 5 | Refund not received, partial refund, merchant charge |
| Credit Card | 5 | Unauthorized transaction, lost card, EMI conversion |
| Loan | 5 | Foreclosure charges, missed EMI, prepayment |
| Security and Fraud | 5 | OTP scam, phishing link, fake caller |
| Ambiguous | 1 | Unfamiliar debit with reversal demand |

Each case includes:
- `question` — the customer's message
- `expected_points` — what the response should cover (4 per case)
- `forbidden_claims` — what the response must not say (2 per case)

### Intent classification test cases (30 cases)

Categories: Payment, Refund, Loan, Credit Card — including edge cases like credit card payment (classified as Payment) and fraud scenarios.

---

## How the Modelfiles work

### `Modelfile` (Banking Assistant)

```dockerfile
FROM mistral:latest

SYSTEM """[Banking-specific system prompt with safety rules]"""

PARAMETER temperature 0         # Deterministic output
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 700       # Enough tokens for detailed responses
```

The system prompt enforces:
- No invented policies, charges, fees, or timelines
- No guaranteed outcomes
- Never ask for OTP, CVV, PIN, or passwords
- Direct customers to authenticated official channels
- Domain-specific guidance for payments, refunds, loans, fraud

### `Modelfile.classifier` (Intent Classifier)

```dockerfile
FROM mistral:latest

SYSTEM """[Classification rules with allowed categories]"""

PARAMETER temperature 0
PARAMETER num_predict 10        # Only needs 1-2 words output
```

The classifier uses strict output rules: return exactly one category name, no explanation.

---

## Customizing the models

### Modify behavior

Edit the `SYSTEM` section in `Modelfile` or `Modelfile.classifier`, then rebuild:

```powershell
ollama create banking-assistant -f Modelfile
```

### Change base model

Replace `FROM mistral:latest` with another model:

```dockerfile
FROM llama3.1:latest
```

### Tune parameters

Adjust `PARAMETER` values. Key parameters:

| Parameter | Effect |
|---|---|
| `temperature 0` | Deterministic, consistent responses |
| `temperature 0.7` | More creative, varied responses |
| `top_p 0.9` | Nucleus sampling threshold |
| `num_predict 700` | Maximum output tokens |

---

## Troubleshooting

### "Ollama connection error"

- Ensure Ollama is running (check system tray on Windows)
- Run `ollama list` to test
- Start with `ollama serve` in a separate terminal if needed

### "Model not found"

Run `python src/create_models.py` to build the custom models.

Or manually:

```powershell
ollama create banking-assistant -f Modelfile
ollama create banking-classifier -f Modelfile.classifier
```

### "Module not found" when running scripts

Run from the project root:

```powershell
cd "C:\Users\Rishav\Disk\Codes\LLM Projects\Fine_Tuning"
python src/evaluate.py
```

---

## Hardware

| Item | Configuration |
|---|---|
| OS | Windows / PowerShell |
| Python | 3.13+ |
| Ollama | 0.24.0+ |
| GPU | NVIDIA GeForce RTX 3050 (4 GB VRAM) |

The 4 GB VRAM is sufficient for running quantized inference models through Ollama. It is **not** sufficient for traditional LoRA/QLoRA weight training of 7B models. The Modelfile approach used here works within this hardware constraint.

---

## Future: True weight-based fine-tuning

If you want to go beyond Modelfile customization and actually modify model weights:

1. **Train on cloud GPU** — Use Google Colab (free T4 GPU) with [Unsloth](https://github.com/unslothai/unsloth) for fast QLoRA training
2. **Export to GGUF** — `model.save_pretrained_gguf("output", tokenizer)`
3. **Update Modelfile** — Change `FROM mistral:latest` to `FROM ./output/your_model.gguf`
4. **Rebuild** — `ollama create banking-assistant -f Modelfile`

This would teach the model new response patterns at the weight level, not just through prompting.

---

## What was removed and why (detailed)

| Removed file/directory | Lines | Why removed |
|---|---|---|
| `evaluations/evaluator.py` | 200 | Replaced by `src/evaluate.py` — same functionality in 1/3 the code |
| `evaluations/metrics.py` | 346 | Risk detection and coverage merged into `src/evaluate.py` |
| `evaluations/llm_judge.py` | 1068 | LLM-as-a-Judge was never calibrated and blocked progress |
| `evaluations/local_evaluator/runner.py` | 714 | Semantic evaluation runner — unnecessary for fine-tuning |
| `evaluations/local_evaluator/coverage.py` | 115 | Lexical coverage — merged into evaluate.py |
| `evaluations/local_evaluator/criteria.py` | 79 | Criterion ID generator — not needed |
| `evaluations/local_evaluator/safety.py` | 208 | Safety screening — simplified in evaluate.py |
| `evaluations/local_evaluator/semantic_assist.py` | 515 | Ollama-based semantic evaluator — expensive and slow |
| `experiments/prompt_comparison.py` | 203 | Prompt experiments — useful for research, not for fine-tuning |
| `experiments/parameter_experiment.py` | 85 | Parameter sweep — one-time experiment, results already captured |
| `experiments/compare_manual_vs_judge.py` | 278 | Judge calibration — the judge was never reliable enough |
| `experiments/create_manual_review.py` | 124 | Manual review template generator — not needed |
| `experiments/create_training_candidates.py` | 99 | Training candidate generator — never used |
| `src/prompt_lab.py` | 121 | Replaced by `src/classify.py` with comparison support |
| `prompts/*.txt` | 4 files | Empty prompt template files |
| `backup_before_local_evaluator_*` | ~30 files | Old backup directory |
| `results/*.csv` | 10+ files | Stale outputs regenerated by new scripts |

**Total removed:** ~3800+ lines of code, ~50+ files

**Total remaining:** ~550 lines of focused, working code across 4 scripts + 2 Modelfiles + 2 datasets
