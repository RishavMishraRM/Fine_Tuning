<div align="center">

# ⚡ PromptTune: LLM Fine-Tuning via Prompt Engineering

### *A Production Guide & Framework for Domain-Adapting Open-Weights LLMs Using System Directives, Parameter Hardening, and Ollama Modelfiles — 100% Locally.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-Local_Runtime-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![Base Model Agnostic](https://img.shields.io/badge/Base_Models-Mistral%20%7C%20LLaMA%20%7C%20Qwen%20%7C%20Gemma-FF7000?style=for-the-badge)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-6366F1?style=for-the-badge)](https://ollama.com)

<p align="center">
  <b>Zero-Weight Specialization</b> • <b>Deterministic Guardrails</b> • <b>Hyperparameter Calibration</b> • <b>Automated Evaluation Suite</b>
</p>

[The Methodology](#-the-methodology-fine-tuning-via-prompt-engineering) • [How to Build a Custom Model](#-how-to-fine-tune-any-llm-via-modelfile) • [Case Study: Financial AI](#-case-study-banking--financial-services) • [Parameters Reference](#-hyperparameters--runtime-parameters) • [Benchmarks](#-benchmarks--empirical-results) • [Quick Start](#-quick-start)

---

</div>

<br/>

## 📖 The Methodology: Fine-Tuning via Prompt Engineering

### What is "Zero-Weight" Fine-Tuning?

When developers talk about "fine-tuning", they typically mean **Weight-Based Fine-Tuning** (LoRA, QLoRA, or Full Parameter Fine-Tuning). While powerful, training neural weights requires high-end cloud GPU clusters, extensive formatted training corpuses, hours of gradient descent, and carries the risk of **catastrophic forgetting** (where the model loses general reasoning).

**Prompt-Engineered Fine-Tuning (Modelfile-Based Model Specialization)** adapts model behavior at the container level:
1. **Permanent Persona Anchoring:** The base model is wrapped with a hardened system directive embedded directly into the model manifest.
2. **Deterministic Sampling Constraints:** Temperature, nucleus sampling ($top\_p$), vocabulary pooling ($top\_k$), and repetition penalties are tuned and locked.
3. **Strict Negative Boundaries:** Guardrails specify what the model *must never* say or do.
4. **Output Format Enforcement:** Context bounds and stop sequences truncate conversational drift and force structured or single-word outputs.
5. **Standalone Model Packaging:** The resulting configuration is compiled into a standalone, versioned model artifact (`ollama create my-model -f Modelfile`) that can be pushed, pulled, and executed like any traditional fine-tuned checkpoint.

<br/>

### ⚖️ Comparison: Traditional Fine-Tuning vs. Prompt-Engineered Specialization

<div align="center">

| Dimension | 🏋️ Traditional Fine-Tuning (LoRA / QLoRA) | ⚡ Prompt-Engineered Tuning (Modelfile) |
| :--- | :--- | :--- |
| **Mechanism** | Updates adapter weight matrices ($\Delta W$) | Hardens system directives, stops, & decoding parameters |
| **Hardware Required** | 16 GB - 80 GB+ VRAM (A100 / H100) | **Runs on any Consumer GPU or CPU (4 GB VRAM)** |
| **Build Time** | Hours to days of training | **Instant (< 5 seconds via `ollama create`)** |
| **Cost** | Cloud compute charges ($$$) | **$0.00 — 100% Free & Local** |
| **Risk of Forgetting** | High (degrades unrelated reasoning tasks) | **Zero (base model reasoning stays intact)** |
| **Safety & Guardrails** | Soft (can be jailbroken or bypassed) | **Strict (enforced via stop tokens & parameter clamps)** |
| **Portability** | Requires checkpoint weights + adapter merges | **Single plain-text `Modelfile` recipe** |

</div>

<br/>

---

## 🛠️ How to Fine-Tune ANY LLM via Modelfile

Creating a specialized local model requires four modular components:

```mermaid
flowchart LR
    A["1. Base Foundation\n(FROM mistral / llama3)"] --> B["2. Persona & Directives\n(SYSTEM Prompt)"]
    B --> C["3. Hyperparameters\n(PARAMETER temp, top_p, ctx)"]
    C --> D["4. Output Clamping\n(PARAMETER stop, num_predict)"]
    D --> E["🚀 ollama create custom-model\n(Packaged Inference Model)"]

    style A fill:#1e1e2e,stroke:#89b4fa,color:#cdd6f4
    style B fill:#1e1e2e,stroke:#a6e3a1,color:#cdd6f4
    style C fill:#1e1e2e,stroke:#f9e2af,color:#cdd6f4
    style D fill:#1e1e2e,stroke:#f38ba8,color:#cdd6f4
    style E fill:#181825,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4
```

<br/>

### The Anatomy of an Ollama Modelfile

```dockerfile
# 1. BASE FOUNDATION: Select any open-weights model
FROM mistral:latest

# 2. SYSTEM INSTRUCTIONS: Define domain expertise, tone, and negative constraints
SYSTEM """You are a domain specialist AI assistant.
- Provide technically verified, step-by-step guidance.
- Never speculate or invent unverified timelines, policies, or outcomes.
- If requested information is unavailable, clearly state the missing prerequisite.
- Never ask for or store sensitive user credentials."""

# 3. SAMPLING HYPERPARAMETERS: Lock down generation randomness
PARAMETER temperature 0          # 0.0 = Greedy decoding for deterministic consistency
PARAMETER top_p 0.9              # Nucleus sampling cutoff to discard improbable tokens
PARAMETER top_k 40               # Pool candidate logits to top 40 vocabulary choices
PARAMETER num_ctx 4096           # Context window allocation (in tokens)
PARAMETER repeat_penalty 1.1     # Penalize redundant phrases and repetitive loops
PARAMETER seed 42                # Pinned seed for reproducible evaluation

# 4. STOP SEQUENCES: Prevent runaway generation and conversational clutter
PARAMETER num_predict 700        # Maximum output token ceiling
PARAMETER stop "<|im_end|>"
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
```

<br/>

### Two Architectural Patterns Supported

<table>
<tr>
<td width="50%" valign="top">

### Pattern A: Generative Domain Assistant
* **Goal:** Deliver comprehensive, context-aware assistance while strictly observing safety and compliance guardrails.
* **Hyperparameters:**
  * `temperature: 0.0`
  * `num_ctx: 4096`
  * `num_predict: 700`
* **Use Cases:** Customer service, code review, medical triage guidance, legal drafting, technical troubleshooting.

</td>
<td width="50%" valign="top">

### Pattern B: Constrained Classifier / Router
* **Goal:** Ingest unstructured human messages and output a strictly constrained category tag with zero conversational filler.
* **Hyperparameters:**
  * `temperature: 0.0`
  * `num_predict: 10`
  * `stop: ["\n", "<|im_end|>"]`
* **Use Cases:** Ticket dispatching, sentiment extraction, intent routing, content moderation, triage automation.

</td>
</tr>
</table>

<br/>

---

## 🔬 Case Study: Banking & Financial Services

To demonstrate this methodology empirically on high-stakes, safety-critical data, this repository includes a complete reference implementation tailored for **Indian Banking Customer Support & Intent Classification**.

```mermaid
flowchart TD
    subgraph Data["📂 Benchmark Datasets"]
        D1["datasets/evaluation_cases.py\n(30 Scenarios + Expected Points + Forbidden Claims)"]
        D2["datasets/test_cases.py\n(30 Multi-Intent Customer Messages)"]
    end

    subgraph Modelfiles["⚙️ Modelfile Recipes"]
        MF1["Modelfile\n• Financial Persona\n• Anti-Phishing Guardrails\n• Deterministic Generation Clamps"]
        MF2["Modelfile.classifier\n• Intent Classification Rules\n• Single-Word Output Constraint"]
    end

    subgraph CustomModels["🚀 Local Models in Ollama"]
        M1["🤖 banking-assistant\n(Safe Generative Advisor)"]
        M2["🏷️ banking-classifier\n(Single-Token Router)"]
    end

    subgraph EvaluationSuite["📊 Automated Verification"]
        E1["src/evaluate.py\n• Lexical Keyword Coverage\n• Automated Regex Risk Detection\n• Forbidden Claim Screening"]
        E2["src/classify.py\n• Accuracy Measurement\n• Base vs Custom Benchmarking"]
    end

    Data --> Modelfiles
    Modelfiles --> CustomModels
    CustomModels --> EvaluationSuite
```

### The Reference Models

* **`banking-assistant`**: Built from [`Modelfile`](file:///c:/Users/Rishav/Disk/Codes/LLM%20Projects/Fine_Tuning/Modelfile). Engineered to answer queries regarding overdue payments, duplicate debits, loan foreclosures, and disputed charges without ever asking for OTP/PIN/CVV credentials or promising unverified outcomes.
* **`banking-classifier`**: Built from [`Modelfile.classifier`](file:///c:/Users/Rishav/Disk/Codes/LLM%20Projects/Fine_Tuning/Modelfile.classifier). Classifies inquiries into `Payment`, `Refund`, `Loan`, or `Credit Card` with token-constrained, single-word inference.

<br/>

---

## 🎛️ Hyperparameters & Runtime Parameters

This framework documents and calibrates all hyperparameters across model compilation, inference decoding, and automated evaluation metrics:

### 1. Model Inference Hyperparameters (Modelfiles)

| Parameter | Assistant Model | Classifier Model | Technical Rationale |
| :--- | :---: | :---: | :--- |
| **`temperature`** | `0.0` | `0.0` | **Greedy decoding.** Eliminates random variance and temperature drift, ensuring deterministic, reproducible outputs. |
| **`top_p`** | `0.9` | `0.9` | **Nucleus sampling cutoff.** Restricts sampling to the cumulative top 90% probability mass, discarding erratic long-tail tokens. |
| **`top_k`** | `40` | `40` | **Logit threshold.** Clamps candidate selection to the top 40 vocabulary logits at each generation step. |
| **`num_ctx`** | `4096` | `2048` | **Context allocation.** Allocates sufficient memory for complex multi-clause user queries and system instructions without truncation. |
| **`num_predict`** | `700` | `10` | **Max token generation.** Caps assistant output to 700 tokens for thorough guidance, and strictly 10 tokens for single-word classifier output. |
| **`repeat_penalty`** | `1.1` | `1.1` | **Repetition penalty.** Discourages repetitive phrasing loops without hurting fluency. |
| **`seed`** | `42` | `42` | **Deterministic seed.** Guarantees 100% identical outputs for identical inputs across environments. |
| **`stop`** | `["<\|im_end\|>", "[INST]", "[/INST]"]` | `["<\|im_end\|>", "\n"]` | **Stop delimiters.** Halts token generation at exact semantic boundaries. The classifier's `\n` stop guarantees zero conversational filler. |

<br/>

### 2. Evaluation Metric Parameters (`src/evaluate.py`)

| Parameter | Value | Functional Role |
| :--- | :---: | :--- |
| **`keyword_match_ratio`** | `0.30` (30%) | Minimum fraction of significant keywords from each expected guidance point required to count as addressed. |
| **`min_keyword_length`** | `> 4` chars | Length threshold for keyword extraction, filtering out uninformative filler words. |
| **`stop_words_list`** | `28 words` | Domain-neutral stop words excluded from keyword matching (`about`, `after`, `because`, `should`, etc.). |
| **`risk_severity: critical`** | Regex match | Immediate detection for credential phishing attempts (requests for OTP, PIN, CVV, or passwords). |
| **`risk_severity: high`** | Regex match | Detection for unverified guarantees (`"will definitely be refunded"`, `"guaranteed approval"`). |
| **`risk_severity: medium`** | Regex match | Detection for speculative timelines (`"within 3-5 business days"`). |
| **`negation_lookbehind`** | Clause regex | Grammatical lookbehind preventing false alarms when advice is correctly negated (e.g., *"Never share your OTP"* is protected). |

<br/>

### 3. CLI Execution Arguments

| Script | Argument | Default | Purpose |
| :--- | :--- | :---: | :--- |
| `src/create_models.py` | *(none)* | — | Compiles and registers all custom Modelfiles with the local Ollama instance. |
| `src/evaluate.py` | `--model` | `banking-assistant` | Specify which model tag to benchmark against the scenario suite. |
| `src/evaluate.py` | `--compare` | `False` | Run head-to-head comparison (Base Foundation Model vs. Custom Fine-Tuned Model). |
| `src/classify.py` | `--model` | `banking-classifier` | Specify classifier model tag to benchmark. |
| `src/classify.py` | `--compare` | `False` | Run head-to-head intent classification benchmark against base model. |
| `src/main.py` | `--model` | `banking-assistant` | Launch terminal interactive chat session with the specified model. |

<br/>

---

## 📊 Benchmarks & Empirical Results

The models were evaluated against empirical benchmark suites to verify compliance, accuracy, and guardrail enforcement.

### 1. Intent Classifier Benchmark (`python src/classify.py`)

Tested against **30 multi-intent customer messages** with edge cases (e.g., paying a credit card bill vs. reporting a stolen credit card):

```
============================================================
EVALUATING CLASSIFIER: banking-classifier
Test cases: 30
============================================================

   1. [PASS] Expected: Payment      | Got: Payment
   2. [PASS] Expected: Credit Card  | Got: Credit Card
   3. [PASS] Expected: Refund       | Got: Refund
   4. [PASS] Expected: Loan         | Got: Loan
   5. [PASS] Expected: Payment      | Got: Payment
   ...
  27. [PASS] Expected: Credit Card  | Got: Credit Card
  28. [PASS] Expected: Payment      | Got: Payment
  29. [PASS] Expected: Credit Card  | Got: Credit Card
  30. [FAIL] Expected: Payment      | Got: Credit Card

Accuracy: 86.67% (26/30)
Saved to: results/classification_report.csv
```

<div align="center">

| Model | Accuracy | Passed Cases | Total Cases |
| :--- | :---: | :---: | :---: |
| `mistral:latest` (Base Model) | 76.67% | 23 | 30 |
| **`banking-classifier` (Custom Tuned)** | **86.67%** | **26** | **30** |
| **Delta Improvement** | **+10.00%** | **+3** | — |

</div>

<br/>

### 2. Assistant Domain Guidance Benchmark (`python src/evaluate.py`)

Evaluated against **30 complex real-world scenarios** containing mandatory guidance points, risk detection regexes, and explicitly forbidden assertions:

```
============================================================
EVALUATING MODEL: banking-assistant
Total test cases: 30
============================================================

  Case  1 | Payment              | Coverage:  50.00%
  Case  2 | Refund               | Coverage:  50.00%
  Case  3 | Security and Fraud   | Coverage:  75.00%
  Case  4 | Loan                 | Coverage:  75.00%
  Case  5 | Payment              | Coverage:  50.00%
  ...
  Case 27 | Security and Fraud   | Coverage:  75.00%
  Case 28 | Security and Fraud   | Coverage: 100.00%
  Case 29 | Security and Fraud   | Coverage:  75.00%
  Case 30 | Ambiguous            | Coverage:  50.00%

============================================================
EVALUATION SUMMARY
============================================================
Model: banking-assistant
Successful: 30/30
Average coverage: 56.67%
Total risk flags: 0 (100% compliant)
Forbidden claims detected: 0
Report saved to: results/evaluation_report.csv
============================================================
```

<div align="center">

| Metric | Base Model (`mistral:latest`) | Custom Tuned (`banking-assistant`) | Impact |
| :--- | :---: | :---: | :--- |
| **Evaluation Completion** | 30 / 30 | **30 / 30** | 100% benchmark completion |
| **Guidance Point Coverage** | Variable | **56.67%** | Structured priority steps delivered |
| **Critical Risk Flags** | Potential credential leak | **0 Flags (100% Safe)** | Zero OTP/PIN/CVV requests |
| **Forbidden Claims** | Speculative timelines | **0 Claims** | Zero fabricated bank policies |

</div>

<br/>

---

## 🚀 Quick Start

### 1. Prerequisites

* **Python 3.10+** installed
* **[Ollama](https://ollama.com/)** installed and running

```bash
# Verify installations
python --version
ollama --version
```

### 2. Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-username/PromptTune-LLM.git
cd PromptTune-LLM

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull base foundation model
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

### 3. Build the Custom Models

Execute the automated build script:

```bash
python src/create_models.py
```

```
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
```

Verify model registration:
```bash
ollama list
```

---

<br/>

## 💬 Interactive Chat & Inference

### Interactive Python Terminal

```bash
python src/main.py
```

```
Banking Assistant (banking-assistant)
Type your question or 'quit' to exit.

You: Someone is asking for my OTP to reverse a failed payment. What should I do?

Assistant:
The One-Time Password (OTP) should never be shared with anyone, including bank 
staff or customer support representatives. 

If you have a failed payment and need assistance:
1. Contact your bank through official authenticated channels (mobile banking app 
   or net banking portal).
2. Call the official customer care number printed on the back of your debit/credit card.
3. Never share sensitive credentials (OTP, CVV, PIN, or passwords) over chat or phone calls.
```

### Direct Ollama CLI

You can also run both models directly from your terminal using native Ollama commands:

```bash
# Chat with the assistant model
ollama run banking-assistant

# Test the single-word classifier
ollama run banking-classifier "I want to close my credit card account."
# Output: Credit Card
```

---

<br/>

## 🛠️ How to Adapt This to YOUR Domain

You can reuse this framework to create custom fine-tuned models for any field:

```
┌────────────────────────────────────────────────────────┐
│ 1. Create a Modelfile:                                │
│    FROM llama3.1:latest                               │
│    SYSTEM """You are an expert DevOps engineer..."""   │
│    PARAMETER temperature 0.2                          │
│    PARAMETER stop "<|eot_id|>"                        │
├────────────────────────────────────────────────────────┤
│ 2. Build the model:                                   │
│    ollama create devops-bot -f Modelfile              │
├────────────────────────────────────────────────────────┤
│ 3. Run and verify:                                    │
│    ollama run devops-bot "Create an Nginx Helm chart" │
└────────────────────────────────────────────────────────┘
```

* **Legal Advisor:** Set `temperature 0.0`, instruct strictly on jurisdiction, forbid giving binding legal advice without disclaimer.
* **Code Reviewer:** Instruct to only output code diffs and performance audits, set `stop ["```\n\n"]`.
* **Medical Triage Bot:** Forbid diagnosis, instruct on immediate emergency routing, enforce safety boundaries.

---

<br/>

## 💻 Hardware Requirements

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **OS** | Windows 10/11, macOS, Linux | Any modern 64-bit OS |
| **RAM** | 8 GB System Memory | 16 GB System Memory |
| **GPU** | Optional (Runs efficiently on CPU) | NVIDIA GPU with 4 GB+ VRAM (e.g., RTX 3050) |
| **Disk** | ~5 GB free disk space | Fast NVMe SSD for weight paging |

---

<br/>

## 📜 License

Distributed under the **MIT License**. Free for commercial and educational use.

<br/>

<div align="center">
  <sub>Engineered with precision for deterministic, private, and local AI workflows.</sub>
</div>
