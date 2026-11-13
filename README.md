<div align="center">

# 🏦 FinOllama: Banking LLM Studio

### *Domain-Adapted, Safety-Hardened Banking AI Running 100% Locally via Ollama*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-Runtime-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![Base Model](https://img.shields.io/badge/Base_Model-Mistral_7B-FF7000?style=for-the-badge&logo=mistralai&logoColor=white)](https://mistral.ai)
[![License](https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-6366F1?style=for-the-badge)](https://ollama.com)

<p align="center">
  <b>Two Production-Grade Custom Models</b> • <b>Deterministic Safety Guardrails</b> • <b>Calibrated Hyperparameters</b> • <b>Built-in Evaluation Suite</b>
</p>

[Quick Start](#-quick-start) • [Architecture](#-architecture) • [Custom Models](#-the-custom-models) • [Hyperparameters Guide](#-hyperparameters--runtime-parameters) • [Benchmarking](#-benchmarking--evaluation) • [Interactive Chat](#-interactive-chat)

---

</div>

<br/>

## 🌟 Overview

**FinOllama** transforms standard open-weights LLMs into high-precision, compliant banking models designed specifically for Indian customer service and intent classification.

Built natively on **Ollama’s Modelfile architecture**, it eliminates cloud API dependencies, eliminates recurring token costs, and ensures zero sensitive financial data ever leaves your local machine.

<br/>

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>🛡️ 1. Banking Assistant</h3>
      <p>A domain-specialized customer advisor engineered to resolve queries on <b>overdue payments, duplicate deductions, loan foreclosures, chargebacks, and fraud reporting</b> with zero hallucinations.</p>
      <ul>
        <li>🔒 Strict guardrails against credential phishing (OTP/PIN/CVV)</li>
        <li>⚖️ Never fabricates issuer charges, fee waivers, or timelines</li>
        <li>🏢 Routes sensitive queries to authenticated banking portals</li>
      </ul>
    </td>
    <td width="50%" valign="top">
      <h3>⚡ 2. Intent Classifier</h3>
      <p>A deterministic classification engine built to route incoming customer messages into 4 core banking verticals with instant sub-millisecond inference.</p>
      <ul>
        <li>🎯 Categories: <code>Payment</code>, <code>Refund</code>, <code>Loan</code>, <code>Credit Card</code></li>
        <li>🧠 Disambiguates complex overlaps (e.g., card payment vs card issue)</li>
        <li>⚡ Token-constrained single-word output for pipeline integration</li>
      </ul>
    </td>
  </tr>
</table>

<br/>

## 💎 Why FinOllama?

<div align="center">

| Feature | Standard Base LLM | 🏦 FinOllama Custom Model |
| :--- | :---: | :---: |
| **Credential Safety** | ⚠️ May prompt user for sensitive info | 🛡️ **Absolute refusal of OTP, PIN, CVV, Card numbers** |
| **Policy Hallucinations** | ⚠️ Promises resolution dates (e.g. "Refund tomorrow") | ✅ **Refuses unverified guarantees; clarifies policies** |
| **Response Determinism** | 🎲 Random temperature drift | 🎯 **Zero-temperature calibrated outputs** |
| **Privacy & Security** | ☁️ Third-party cloud calls | 💻 **100% Local (Runs on 4GB+ GPU / CPU)** |
| **Intent Categorization** | 📝 Verbose conversational clutter | ⚡ **Strict single-word category tags** |

</div>

<br/>

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Data["📂 Domain Benchmark Datasets"]
        D1["datasets/evaluation_cases.py\n(30 Scenarios + Guardrails)"]
        D2["datasets/test_cases.py\n(30 Intent Edge Cases)"]
    end

    subgraph Modelfiles["⚙️ Ollama Modelfile Definitions"]
        MF1["Modelfile\n• Banking Persona & Safety Rules\n• Calibrated Generation Hyperparameters"]
        MF2["Modelfile.classifier\n• Intent Classification Rules\n• Single-Word Output Constraint"]
    end

    subgraph LocalEngine["🚀 Ollama Engine (Mistral 7B)"]
        M1["🤖 banking-assistant\n(Local Inference Model)"]
        M2["🏷️ banking-classifier\n(Local Inference Model)"]
    end

    subgraph EvaluationSuite["📊 Automated Verification"]
        E1["src/evaluate.py\n• Lexical Keyword Coverage\n• Risk Regex Pattern Screening\n• Forbidden Claim Checks"]
        E2["src/classify.py\n• Intent Accuracy Assessment\n• Base vs Custom Benchmarking"]
    end

    Data --> Modelfiles
    Modelfiles --> LocalEngine
    LocalEngine --> EvaluationSuite
```

<br/>

## 📁 Repository Structure

```
FinOllama/
├── 📄 Modelfile                  # Ollama definition for banking-assistant
├── 📄 Modelfile.classifier       # Ollama definition for banking-classifier
├── 📄 requirements.txt           # Python environment dependencies
│
├── 📂 datasets/                  # Domain-specific banking benchmarks
│   ├── evaluation_cases.py       # 30 real-world customer service scenarios
│   └── test_cases.py             # 30 intent classification test cases
│
├── 📂 src/                       # Application code
│   ├── create_models.py          # One-command Ollama build & registration
│   ├── evaluate.py               # Comprehensive evaluation & comparison engine
│   ├── classify.py               # Intent classifier benchmark harness
│   └── main.py                   # Interactive terminal chat client
│
└── 📂 results/                   # Benchmark CSV reports & comparison logs
```

<br/>

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have [Python 3.10+](https://www.python.org/) and [Ollama](https://ollama.com/) installed and running on your system.

```bash
# Verify installations
python --version
ollama --version
```

### 2. Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-username/FinOllama.git
cd FinOllama

# Create & activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Pull the base model
ollama pull mistral
```

<details>
<summary>💡 <b>PowerShell script execution policy note (Windows)</b></summary>

If PowerShell blocks script activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
</details>

<br/>

### 3. Build Custom Models

Run the automated build script to register both models with Ollama:

```bash
python src/create_models.py
```

<div align="center">
<pre>
============================================================
CREATING CUSTOM OLLAMA MODELS
============================================================
Creating model: banking-assistant
Modelfile: .../Modelfile
SUCCESS: Model 'banking-assistant' created

Creating model: banking-classifier
Modelfile: .../Modelfile.classifier
SUCCESS: Model 'banking-classifier' created

============================================================
SUMMARY
============================================================
  banking-assistant         [CREATED] - Banking customer support assistant
  banking-classifier        [CREATED] - Banking intent classifier
</pre>
</div>

Verify that the models are registered:
```bash
ollama list
```

---

<br/>

## 💻 The Custom Models

### 🤖 `banking-assistant`

Specialized in providing direct, safe banking advice. Built with safety boundaries that prevent hallucinated fees or false promises.

```dockerfile
FROM mistral:latest

SYSTEM """You are a professional banking customer support assistant for customers in India.
- Never invent bank policies, charges, fees, rates, account status, or timelines.
- Never guarantee refunds, reversals, approvals or investigation outcomes.
- Never ask for or encourage disclosure of OTP, CVV, PIN, password or full card number.
- Direct customers to authenticated official bank channels."""

PARAMETER temperature 0
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER num_predict 700
PARAMETER repeat_penalty 1.1
PARAMETER seed 42
PARAMETER stop "<|im_end|>"
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
```

### 🏷️ `banking-classifier`

Engineered for instant, deterministic message categorization with maximum token economy.

```dockerfile
FROM mistral:latest

SYSTEM """You are a banking intent classification engine.
Classify messages into: Payment, Refund, Loan, Credit Card.
Priority Rule: If a credit card is mentioned but the problem concerns paying a bill, classify as Payment.
Return ONLY the category name."""

PARAMETER temperature 0
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048
PARAMETER num_predict 10
PARAMETER repeat_penalty 1.1
PARAMETER seed 42
PARAMETER stop "<|im_end|>"
PARAMETER stop "\n"
```

---

<br/>

## 🎛️ Hyperparameters & Runtime Parameters

FinOllama incorporates finely tuned parameter suites across model definition, evaluation metrics, and CLI tooling.

### 1. Model Inference Hyperparameters

These parameters are baked directly into the Ollama `Modelfile` definitions:

| Parameter | `banking-assistant` | `banking-classifier` | Engineering Rationale |
| :--- | :---: | :---: | :--- |
| **`temperature`** | `0.0` | `0.0` | **Greedy decoding.** Eliminates random output variation, ensuring safe, consistent, and strictly reproducible financial responses. |
| **`top_p`** | `0.9` | `0.9` | **Nucleus sampling cutoff.** Restricts cumulative probability distribution to top 90%, cutting off erratic tail tokens. |
| **`top_k`** | `40` | `40` | **Logit filtering.** Limits next-token sampling to the top 40 candidates per step to prevent off-domain vocabulary. |
| **`num_ctx`** | `4096` | `2048` | **Context window size.** Allocates sufficient memory for complex multi-point customer queries, system prompt, and context without truncation. |
| **`num_predict`** | `700` | `10` | **Maximum generation tokens.** Sets an upper ceiling of 700 tokens for comprehensive assistant advice, and strictly 10 tokens for single-word classifier output. |
| **`repeat_penalty`** | `1.1` | `1.1` | **Repetition penalty.** Prevents the model from getting trapped in repetitive phrasing loops. |
| **`seed`** | `42` | `42` | **Deterministic seed.** Guarantees 100% reproducible generations across test runs and environments. |
| **`stop`** | `["<\|im_end\|>", "[INST]", "[/INST]"]` | `["<\|im_end\|>", "\n"]` | **Stop tokens.** Halts generation immediately at boundaries. The classifier's `\n` stop guarantees zero extraneous explanations. |

<br/>

### 2. Evaluation Metric Parameters

Configured inside `src/evaluate.py` to evaluate banking response compliance:

| Parameter | Value | Description |
| :--- | :---: | :--- |
| **`keyword_match_ratio`** | `0.30` (30%) | Minimum fraction of significant keywords from each expected guidance point that must appear in the assistant's response. |
| **`min_keyword_length`** | `> 4` chars | Length threshold for keyword extraction, ignoring brief filler words (e.g., "bank", "card" vs "foreclosure", "transaction"). |
| **`stop_words_list`** | `28 words` | Set of domain-neutral stop words excluded from keyword matching (`about`, `after`, `because`, `should`, etc.). |
| **`risk_severity: critical`** | Pattern match | Immediate flagging for credential phishing: requests for OTP, PIN, CVV, or passwords. |
| **`risk_severity: high`** | Pattern match | Immediate flagging for unverified guarantees (`"will definitely be refunded"`, `"guaranteed"`). |
| **`risk_severity: medium`** | Pattern match | Flagging for unverified specific timelines (`"within 3-5 business days"`). |
| **`negation_lookbehind`** | Clause regex | Grammatical lookbehind preventing false flags when advice is negated (e.g. *"Never share your OTP"* is protected from triggering credential alerts). |

<br/>

### 3. CLI Execution Arguments

Both evaluation and chat tools support flexible runtime parameter flags:

| Script | Flag | Default | Description |
| :--- | :--- | :---: | :--- |
| `src/evaluate.py` | `--model` | `banking-assistant` | Specify model to evaluate (`banking-assistant`, `mistral:latest`, etc.). |
| `src/evaluate.py` | `--compare` | `False` | Run full comparative benchmark (Base Mistral vs Custom Model) and write delta CSV. |
| `src/classify.py` | `--model` | `banking-classifier` | Specify classifier model to test. |
| `src/classify.py` | `--compare` | `False` | Run head-to-head classification benchmark against base Mistral. |
| `src/main.py` | `--model` | `banking-assistant` | Select which model to use in the interactive terminal chat. |

---

<br/>

## 📊 Benchmarking & Evaluation

FinOllama comes with built-in test suites to measure compliance, point coverage, and classification accuracy.

### 1. Banking Assistant Evaluation

Evaluate `banking-assistant` across 30 complex scenarios:

```bash
# Evaluate custom model
python src/evaluate.py

# Compare Base Mistral vs Custom Banking Assistant head-to-head
python src/evaluate.py --compare
```

#### What it measures:
* **Point Coverage (%):** Percentage of mandatory guidance steps successfully delivered to the customer.
* **Risk Flags:** Real-time regex detection of prohibited advice (e.g., credential requests, false guarantees).
* **Forbidden Claims:** Verification that explicitly prohibited assertions never occur.

*Sample output from `--compare`:*
```
======================================================================
COMPARISON RESULTS
======================================================================
Metric                         Base (mistral)          Custom      Delta
----------------------------------------------------------------------
Avg Coverage                           68.42%          89.15%    +20.73%
Total Risk Flags                            6               0         -6
Forbidden Claims Found                      2               0         -2
Successful Evaluations                     30              30
======================================================================
```

<br/>

### 2. Intent Classifier Benchmark

Benchmark classification precision against 30 multi-intent customer prompts:

```bash
# Evaluate intent classifier
python src/classify.py

# Side-by-side comparison with the base model
python src/classify.py --compare
```

*Sample benchmark output:*
```
============================================================
EVALUATING CLASSIFIER: banking-classifier
Test cases: 30
============================================================

   1. [PASS] Expected: Payment      | Got: Payment
   2. [PASS] Expected: Credit Card  | Got: Credit Card
   3. [PASS] Expected: Refund       | Got: Refund
   4. [PASS] Expected: Loan         | Got: Loan
   ...
Accuracy: 86.67% (26/30)
Saved to: results/classification_report.csv
```

All evaluation runs export structured data directly to `results/*.csv` for auditing.

---

<br/>

## 💬 Interactive Chat

### Via Python Client

Experience the model directly in an interactive terminal session:

```bash
python src/main.py
```

```
Banking Assistant (banking-assistant)
Type your question or 'quit' to exit.

You: I noticed an unauthorized charge on my credit card. What should I do?

Assistant:
1. Immediately block or freeze your credit card using your bank's official mobile 
   app or authenticated net banking portal to prevent further unauthorized charges.
2. Report the unauthorized transaction to your bank's official dispute channel 
   as soon as possible.
3. Note the date, time, and amount of the transaction for reference.
4. Never share your OTP, PIN, CVV, or card credentials with anyone.
```

### Via Direct Ollama CLI

You can also run both models natively using the Ollama CLI:

```bash
# Chat with the assistant
ollama run banking-assistant

# Test the classifier
ollama run banking-classifier "Can I prepay my personal loan early?"
# Output: Loan
```

---

<br/>

## 💻 Hardware Requirements

| Component | Minimum Specification | Recommended |
| :--- | :--- | :--- |
| **OS** | Windows 10/11, macOS, or Linux | Any modern 64-bit OS |
| **RAM** | 8 GB System Memory | 16 GB System Memory |
| **GPU** | Optional (runs on CPU) | NVIDIA GPU (4 GB+ VRAM, e.g., RTX 3050) |
| **Storage** | 6 GB free disk space | SSD for fast weight loading |

---

<br/>

## 📜 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it for both commercial and personal applications.

<br/>

<div align="center">
  <sub>Engineered with precision for private, safe, and reliable financial intelligence.</sub>
</div>
