# AraHalluEval

A hallucination evaluation framework for Arabic LLMs, covering **Generative Question Answering (GQA)** and **Text Summarization** tasks.

**Authors:** Aisha Alansari & Hamzah Luqman — KFUPM / SDAIA-KFUPM Joint Research Center for AI  
**Paper:** [AraHalluEval: A Fine-grained Hallucination Evaluation Framework for Arabic LLMs](https://aclanthology.org/2025.arabicnlp-main.12/) — ArabicNLP 2025  
**arXiv:** [2509.04656](https://arxiv.org/abs/2509.04656)

---

## Repository Contents

```
AraHalluEval/
├── models.py                       # Model registry and loader
├── inference.py                    # Unified response generation (HF / OpenAI / Together AI)
├── run.py                          # Main entry point — runs a model on a task and saves results
├── AraHalluEval_QA.csv             # Annotated QA evaluation data
├── AraHalluEval_Summarization.csv  # Annotated summarization evaluation data
└── TruthfulQA_translated_New.xlsx  # TruthfulQA manually translated to Arabic
```

---

## Code Overview

### `models.py`

Defines `MODEL_MAP`, the central registry mapping short model keys to their HuggingFace, OpenAI, or Together AI identifiers, and exposes `load_model_and_tokenizer()`.

**Supported models:**

| Key | Backend | Model ID |
|---|---|---|
| `Jais-6.7b` | HuggingFace | `inceptionai/jais-family-6p7b` |
| `Noon` | HuggingFace | `Naseej/noon-7b` |
| `Allam` | HuggingFace | `ALLaM-AI/ALLaM-7B-Instruct-preview` |
| `Fanar` | HuggingFace | `QCRI/Fanar-1-9B` |
| `Gemma` | HuggingFace | `google/gemma-3-4b-pt` |
| `llama` | HuggingFace | `meta-llama/Meta-Llama-3-8B` |
| `qwen2.5-7b` | HuggingFace | `Qwen/Qwen2.5-7B` |
| `bloom-7b` | HuggingFace | `bigscience/bloom-7b` |
| `openai:gpt-4o` | OpenAI API | `gpt-4o` |
| `openai:gpt-o3` | OpenAI API | `gpt-o3` |
| `together:deepseek-r1` | Together AI | `deepseek-ai/DeepSeek-R1-0528` |
| `together:deepseek-v3` | Together AI | `deepseek-ai/DeepSeek-V3` |
| `together:qwen-qwq` | Together AI | `Qwen/QwQ-32B` |
| `together:llama4-Maverick` | Together AI | `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` |

`load_model_and_tokenizer(model_key)` returns `(model, tokenizer, device)`. For API-backed models (`openai:*`, `together:*`) it returns `(None, None, None)` since no local loading is needed.

---

### `inference.py`

Contains `get_response(prompt, model, tokenizer, device, model_key, task, debug)` — a single unified function that routes to the correct backend:

- **OpenAI** — calls `openai.ChatCompletion.create` with `temperature=0.0`.
- **Together AI** — calls `together.Complete.create` with `max_tokens=200`, `temperature=0.0`, `repetition_penalty=1.2`.
- **HuggingFace** — tokenizes the prompt, runs `model.generate` with greedy decoding (`do_sample=False`, `temperature=0.0`, `max_new_tokens=200`, `repetition_penalty=1.2`, `pad_token_id=tokenizer.eos_token_id`), then decodes and post-processes the output.

**Post-processing:** For HuggingFace models the function strips the prompt echo and extracts text after Arabic section markers (`الملخص:`, `الجواب:`, `الترجمة:`). For the summarization task it additionally truncates at the first sentence boundary.

---

### `run.py`

The main script. Accepts `--model` and `--task` as CLI arguments, loads the dataset, iterates over rows with a `tqdm` progress bar, calls `get_response` for each, and writes a `results.csv` to a per-model output directory.

**Tasks and their input files / columns / Arabic prompts:**

| Task key | Input CSV | Column used | Prompt |
|---|---|---|---|
| `summarization` | `summ_merged_predictions.csv` | `text` | `لخص النص الآتي في جملة واحدة فقط، وأجب باللغة العربية` |
| `qa` | `QA_merged_predictions.csv` | `question_text` | `اجب على السؤال التالي باللغة العربية` |
| `tru_ar` | `TruthfulQA_confirmed_questions.csv` | `Translated Question` | `اجب على السؤال التالي باللغة العربية` |
| `tru_en` | `TruthfulQA_confirmed_questions.csv` | `Original Question` | `Answer the following question` (English) |

Output is saved to `<task>_<model_key>/results.csv` with a `response` column appended to the original dataframe.

---

## Installation

```bash
git clone https://github.com/aishaalansari57/AraHalluEval.git
cd AraHalluEval
pip install torch transformers huggingface_hub openai together pandas tqdm
```

---

## Setup

Open `models.py` and fill in your credentials before running:

```python
# Uncomment and add your tokens
login(token="YOUR_HF_TOKEN")          # required for gated models (Allam, LLaMA, etc.)
openai.api_key = "YOUR_OPENAI_KEY"    # required for openai:* models
together.api_key = "YOUR_TOGETHER_KEY" # required for together:* models
```

---

## Usage

```bash
python run.py --model <model_key> --task <task>
```

**Examples:**

```bash
# Run Allam on Arabic QA
python run.py --model Allam --task qa

# Run GPT-4o on Arabic summarization
python run.py --model openai:gpt-4o --task summarization

# Run DeepSeek-R1 on Arabic TruthfulQA
python run.py --model together:deepseek-r1 --task tru_ar

# Run LLaMA on English TruthfulQA
python run.py --model llama --task tru_en
```

Results are saved to `./<task>_<model_key>/results.csv`.

---

## Datasets

| File | Description |
|---|---|
| `AraHalluEval_QA.csv` | Manually annotated hallucination labels for LLM outputs on 300 Arabic GQA samples (TyDiQA-GoldP-AR) |
| `AraHalluEval_Summarization.csv` | Manually annotated hallucination labels for LLM outputs on 100 Arabic summarization samples (XLSum) |
| `TruthfulQA_translated_New.xlsx` | 737 TruthfulQA questions with manual Arabic translations for cross-lingual evaluation |

---

## Citation

```bibtex
@inproceedings{alansari-luqman-2025-arahallueval,
    title     = "{A}ra{H}allu{E}val: A Fine-grained Hallucination Evaluation Framework for {A}rabic {LLM}s",
    author    = "Alansari, Aisha and Luqman, Hamzah",
    booktitle = "Proceedings of The Third Arabic Natural Language Processing Conference",
    month     = nov,
    year      = "2025",
    address   = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2025.arabicnlp-main.12/",
    doi       = "10.18653/v1/2025.arabicnlp-main.12",
    pages     = "148--161",
}
```
