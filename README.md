# Role-Based Access Control for Text-to-SQL Benchmark

This repository contains the implementation for evaluating Large Language Models' (LLMs) ability to generate SQL queries while respecting **Role-Based Access Control (RBAC)** constraints. We extend existing Text-to-SQL benchmarks (Spider, BIRD, LiveSQLBench) by introducing user roles with fine-grained table-level permissions, training models to both generate correct SQL and refuse unauthorized queries.

**Key Features**:
- Automatic role generation with table-level permissions using LLMs
- RBAC dataset construction for Spider, BIRD, and LiveSQLBench
- Comprehensive evaluation metrics for security compliance
- Supervised fine-tuning for efficient training (Optional)

## Table of Contents

- [Environment Preparation](#1-environment-preparation)
- [Dataset Preparation](#2-dataset-preparation)
- [Inference with Cloud APIs](#3-inference-with-cloud-apis)
- [Evaluation](#4-evaluation)
- [Fine-tuning (Optional)](#5-fine-tuning-optional)
- [Code Structure](#6-code-structure)

---

## 1. Environment Preparation

We use CUDA 12.1 and **Python 3.10+**. Create a conda environment:

```bash
conda create -n rolesql python=3.10
conda activate rolesql
```

Install dependencies. May take a while to setup.

```bash
pip install -r requirements.txt
```

---

## 2. Dataset Preparation

### 2.1 Base Datasets

Download the base Text-to-SQL datasets:

**Spider**:
``` bash
# 1. Use Spider dataset google link to download: (https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&authuser=0): 
gdown --id 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J -O spider_data.zip

# 2. Then organize your working path:
mkdir -p data/spider/ && unzip spider_data.zip -d data/spider/ && mv data/spider/spider_data/* data/spider/ && rm -rf data/spider/spider_data data/spider/__MACOSX
```

**BIRD** (optional):
```bash
# Follow instructions at https://bird-bench.github.io/
```

**LiveSQLBench** (optional):
```bash
# https://huggingface.co/datasets/birdsql/livesqlbench-base-lite-sqlite
# To prevent data leakage through automated crawling, please request access to the ground truth and test cases by email.
```

### 2.2 Role-Aware Dataset Generation

You have two options to obtain role-augmented datasets:

#### 2.2.1 Option 1: Download Pre-generated Datasets (Recommended)

Download ready-to-use role-aware datasets from Hugging Face:

```bash
# Install huggingface-cli if not already installed
pip install "huggingface-hub<1.0,>=0.34.0"

# Step 1: Download datasets
python -c "from huggingface_hub import snapshot_download; snapshot_download('sharkiefff/RBAC_Text2SQL', repo_type='dataset', local_dir='data/role_datasets_temp')"

# Step 2: Organize files into folders
mkdir -p data/selected/{spider,bird,livesqlbench}
mv data/role_datasets_temp/spider*.json data/selected/spider/
mv data/role_datasets_temp/bird*.json data/selected/bird/
mv data/role_datasets_temp/livesqlbench*.json data/selected/livesqlbench/
rm -rf data/role_datasets_temp
```

#### 2.2.2 Option 2: Generate from None

We provide interactive notebook for generating role-augmented datasets:

**For Spider**:
```bash
# Open the Spider role generation notebook at src/quick_assignment_notebook/spider_role_generation.ipynb

# Follow the notebook instructions to:
# 1. Configure role generation settings
# 2. Generate role assignments
# 3. Create role-aware training datasets
```

**Same For BIRD and LiveSQLBench**:
```bash
jupyter notebook src/quick_assignment_notebook/bird_role_generation.ipynb
```

The generated dataset will have the following format:
```json
{
  "instruction": "System prompt",
  "role": "Data Analyst",
  "tables": ["employees", "departments"],
  "input": "Schema: ...\nQuestion: What is the average salary?",
  "output": "SELECT AVG(salary) FROM employees;",
  "answerable": true,
  "db_id": "company"
}
```
---

## 3. Inference with Cloud APIs

After obtaining role-aware datasets, you can use cloud APIs to generate predictions without local GPU training.

### 3.1: Configure API Keys

Edit in your `.env` file and add your API key(s):

```bash
# Choose one or more providers:
DEEPINFRA_API_KEY=your-deepinfra-key-here    # Recommended: supports many open-source models
DEEPSEEK_API_KEY=sk-your-deepseek-key-here   # Cost-effective option
OPENAI_API_KEY=sk-your-openai-key-here       # For GPT models
ANTHROPIC_API_KEY=sk-ant-your-key-here       # For Claude models
```

### 3.2: Run Prediction

Use the cloud inference script to generate SQL predictions:

```bash
# goes to experiments/scripts/predict_cloud.sh
# modify the Default parameters around line 29 to predict
./experiments/scripts/predict_cloud.sh
```

**Available Providers and Models**:

| Provider | Example Models | API Key Env |
|----------|---------------|-------------|
| `deepinfra` | `google/gemma-3-4b-it`, `google/gemma-3-27b-it` | `DEEPINFRA_API_KEY` |
| `deepseek` | `deepseek-coder`, `deepseek-reasoner` | `DEEPSEEK_API_KEY` |
| `openai` | `gpt-4o-mini`, `gpt-5-mini`, `gpt-5` | `OPENAI_API_KEY` |
| `anthropic` | `claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| `gemini` | `gemini-2.5-flash` | `GEMINI_API_KEY` |

**Common Options**:
- `--max_workers`: Number of concurrent API requests
- `--temperature`: Sampling temperature (default: 0.0 for deterministic output)
- `--max_tokens`: Maximum tokens in response
- `--max_samples`: Max samples from the dataset, leave blank for full
- `--role`: Role-based evaluation flag, set true to use RBAC-data



## 4. Evaluation
You can directly modify corresponding bash instructions in experiments/scripts/eval.sh to run the evaluation.


### 4.1 Evaluating on Spider

Without Role:
```bash
python experiments/eval/evaluation_spider.py \
    --input experiments/output/pred/pred_qwen2.5-coder-7b-bird-role_cleaned.sql \
    --difficulty_json data/selected/spider/spider_dev.json \
    --etype exec \
    --fair_comparison
```

With Role:
```bash
python experiments/eval/evaluation_spider_role.py \
    --input experiments/output/pred/pred_google-gemma-3-4b-it_spider_role.sql \
    --role_json data/selected/spider/spider_dev_with_role.json \
    --etype exec \
    --fair_comparison
```

### 4.2 Evaluating on BIRD

```bash
# find in experiments/scripts/eval.sh
```

### 4.3 Evaluating on LiveSQLBench

```bash
# find in experiments/scripts/eval.sh
```

### 4.4 Evaluation Metrics

The evaluation script computes:
- **EX (Execution Accuracy)**: % of queries producing correct results
- **Answerable Rate**: % of answerable queries the model attempts
- **Correct Refusal Rate**: % of unanswerable queries correctly refused
- **Incorrect Refusal Rate**: % of answerable queries incorrectly refused
- **Violation Rate**: % of unanswerable queries where model generates SQL

---



## 5. Fine-tuning (Optional)

If you want to fine-tune your own models instead of using cloud APIs:

Change the ```model_name_or_path``` in experiments/scripts/train_sft.sh, around line 20; 

Then train using QLoRA on Spider role-aware dataset:

```bash
bash experiments/scripts/train_sft.sh
```

Key configuration in `train_sft.sh`:
```bash
MODEL_PATH="Qwen/Qwen2.5-14B-Instruct"
DATASET_NAME="spider_role_train"
OUTPUT_DIR="outputs/qwen2.5_spider_role"
LORA_RANK=64
LORA_ALPHA=16
NUM_EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE=5e-5
```

**Multi-GPU Training**:
```bash
bash experiments/scripts/train_sft.sh
```

Training uses:
- **4-bit quantization** (QLoRA) for memory efficiency
- **LoRA adapters** (rank=64) for parameter-efficient fine-tuning
- **DeepSpeed ZeRO-3** for distributed training
- **Causal LM objective** with assistant-only loss (prompt masked with `IGNORE_INDEX=-100`)

---



## 6. Code Structure

Below is a concise, up-to-date view of the repository layout. Large data files and log directories are intentionally summarized (not listed).

```
Role-SQL-benchmark/
├── README.md
├── requirements.txt
├── configs/                        # project configuration (paths, prompts, seed)
├── data/                           # datasets and generated role-aware datasets 
├── experiments/                    # training / eval / tooling
│   ├── train/
│   ├── eval/
│   ├── llm_base/
│   ├── data_process/
│   ├── output/
│   └── scripts/
├── src/                            # main code
│   ├── processors/                 # dataset / role generation scripts
│   ├── llm_oracle/                 # LLM role-generation oracle
│   ├── role_parser/                # parsing LLM outputs into roles
│   ├── role_evaluator/             # evaluation utilities and decision logic
│   └── quick_assignment_notebook/  # interactive notebooks for role-gen
├── logs/                           # runtime logs
└── cost_analysis.ipynb             # analysis notebook
```
---