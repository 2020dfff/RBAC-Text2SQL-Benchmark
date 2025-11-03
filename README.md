# Role-Based Access Control for Text-to-SQL Benchmark

This repository contains the implementation for evaluating Large Language Models' (LLMs) ability to generate SQL queries while respecting **Role-Based Access Control (RBAC)** constraints. We extend existing Text-to-SQL benchmarks (Spider, BIRD, LiveSQLBench) by introducing user roles with fine-grained table-level permissions, training models to both generate correct SQL and refuse unauthorized queries.

**Key Features**:
- Automatic role generation with table-level permissions using LLMs
- RBAC dataset construction for Spider, BIRD, and LiveSQLBench
- Comprehensive evaluation metrics for security compliance
- Supervised fine-tuning for efficient training (Optional)

## Table of Contents

- [Environment Preparation](#environment-preparation)
- [Dataset Preparation](#dataset-preparation)
- [Evaluation](#evaluation)
- [Fine-tuning](#fine-tuning)
- [Code Structure](#code-structure)

---

## 1. Environment Preparation

We use CUDA 12.1 and **Python 3.10+**. Create a conda environment:

```bash
conda create -n rolesql python=3.10
conda activate rolesql
```

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



## 4. Evaluation
You can directly modify corresponding bash instructions in experiments/scripts/eval.sh to run the evaluation.


### 4.1 Evaluating on Spider

```bash
python experiments/eval/evaluation_spider_role.py \
    --model_path outputs/qwen2.5_spider_role/checkpoint-best \
    --dataset_file data/spider_role/dev_role.json \
    --database_path spider/database \
    --output_file results/spider_eval.json
```

### 4.2 Evaluating on BIRD

```bash
python experiments/eval/evaluation_bird_role.py \
    --model_path outputs/qwen2.5_bird_role/checkpoint-best \
    --dataset_file data/bird_role/dev_role.json \
    --output_file results/bird_eval.json
```

### 4.3 Evaluating on LiveSQLBench

```bash
python experiments/eval/evaluation_livesqlbench_role.py \
    --model_path outputs/qwen2.5_livesql_role/checkpoint-best \
    --dataset_file data/livesqlbench_role/questions_role.json \
    --output_file results/livesql_eval.json
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

```
Role-SQL-benchmark/
├── configs/
│   ├── prompts.py                    # System/user prompts for role generation
|   ├── seed.py                       # Global random seed
│   └── paths.py                      # Centralized path configuration
├── data/
│   ├── spider_role/                  # Spider role-aware datasets
│   ├── bird_role/                    # BIRD role-aware datasets
│   ├── livesqlbench_role/            # LiveSQLBench role-aware datasets
│   ├── schemas_pg/                   # PostgreSQL schema definitions
│   └── selected/
│       └── dataset_info.json         # Dataset configuration mapping
├── experiments/
│   ├── train/
│   │   ├── train_sft.sh              # Main training script
│   │   └── sft_train.py              # Training entry point
│   ├── eval/
│   │   ├── evaluation_spider_role.py # Spider evaluation
│   │   ├── evaluation_bird_role.py   # BIRD evaluation
│   │   └── evaluation_livesqlbench_role.py
│   └── llm_base/
│       ├── load_tokenizer.py         # Model/tokenizer loading with quantization
│       ├── adapter.py                # LoRA adapter injection via PEFT
│       └── model_trainer.py          # Custom trainer with PeftModelMixin
├── src/
│   ├── processors/
│   │   ├── spider_role_sql_generator.py    # Spider role-aware dataset generation
│   │   ├── bird_role_sql_generator.py      # BIRD role-aware dataset generation
│   │   └── livesqlbench_role_sql_generator.py
│   ├── role_parser/
│   │   └── role_output_parser.py     # LLM output parsing for role extraction
│   ├── db_migration/
│   │   └── postgres_utils.py         # PostgreSQL database utilities
│   ├── evaluator/
│   │   ├── spider_evaluator.py       # Execution-based evaluation
│   │   └── livesqlbench_utils.py     # Evaluation metrics computation
│   └── utils/
│       └── sql_utils.py              # SQL parsing and validation
├── llm_oracle/
│   ├── oracle.py                     # LLM-based role generation oracle
│   └── README.md
├── spider/                           # Original Spider dataset
│   ├── train_spider.json
│   ├── dev.json
│   └── database/
└── requirements.txt                  # Python dependencies
```

---