# Role-Based Access Control for Text-to-SQL Benchmark

This repository contains the implementation for evaluating Large Language Models' (LLMs) ability to generate SQL queries while respecting **Role-Based Access Control (RBAC)** constraints. We extend existing Text-to-SQL benchmarks (Spider, BIRD, LiveSQLBench) by introducing user roles with fine-grained table-level permissions, training models to both generate correct SQL and refuse unauthorized queries.

**Key Features**:
- Automatic role generation with table-level permissions using LLMs
- Permission-aware dataset creation for Spider, BIRD, and LiveSQLBench
- QLoRA-based fine-tuning for efficient training on consumer GPUs
- Comprehensive evaluation metrics for security compliance

---

## Table of Contents

- [Environment Preparation](#environment-preparation)
- [Dataset Preparation](#dataset-preparation)
- [Fine-tuning](#fine-tuning)
- [Evaluation](#evaluation)
- [Code Structure](#code-structure)

---

## Environment Preparation

We use CUDA 12.1 and **Python 3.10+**. Create a conda environment:

```bash
conda create -n role-sql python=3.10
conda activate role-sql
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `torch>=2.0.0`
- `transformers>=4.35.0`
- `peft>=0.7.0` (for LoRA)
- `bitsandbytes>=0.41.0` (for 4-bit quantization)
- `deepspeed>=0.12.0` (for distributed training)
- `psycopg2-binary>=2.9.0` (for PostgreSQL evaluation)

---

## Dataset Preparation

### Base Datasets

Download the base Text-to-SQL datasets:

**Spider**:
``` bash
# go to Spider dataset downloading page: https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&authuser=0
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

### Role-Aware Dataset Generation

You have two options to obtain role-augmented datasets:

#### Option 1: Download Pre-generated Datasets (Recommended)

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

#### Option 2: Generate from Scratch

Generate role-augmented datasets using interactive notebooks:

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

## Inference with Cloud APIs

After obtaining role-aware datasets, you can use cloud APIs to generate predictions without local GPU training.

### Step 1: Configure API Keys

Edit in your `.env` file and add your API key(s):

```bash
# Choose one or more providers:
DEEPINFRA_API_KEY=your-deepinfra-key-here    # Recommended: supports many open-source models
DEEPSEEK_API_KEY=sk-your-deepseek-key-here   # Cost-effective option
OPENAI_API_KEY=sk-your-openai-key-here       # For GPT models
ANTHROPIC_API_KEY=sk-ant-your-key-here       # For Claude models
```

### Step 2: Run Prediction

Use the cloud inference script to generate SQL predictions:

```bash
# goes to experiments/scripts/predict_cloud.sh
# modify the Default parameters around line 29 to predict
```

**Available Providers and Models**:

| Provider | Example Models | API Key Env |
|----------|---------------|-------------|
| `deepinfra` | `google/gemma-3-4b-it`, `google/gemma-3-27b-it` | `DEEPINFRA_API_KEY` |
| `deepseek` | `deepseek-chat`, `deepseek-coder` | `DEEPSEEK_API_KEY` |
| `openai` | `gpt-4o-mini`, `gpt-5-mini` | `OPENAI_API_KEY` |
| `anthropic` | `claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| `gemini` | `gemini-2.5-flash` | `GEMINI_API_KEY` |

**Common Options**:
- `--max_workers`: Number of concurrent API requests
- `--temperature`: Sampling temperature (default: 0.0 for deterministic output)
- `--max_tokens`: Maximum tokens in response



## Evaluation

### Evaluating on Spider

```bash
python experiments/eval/evaluation_spider_role.py \
    --model_path outputs/qwen2.5_spider_role/checkpoint-best \
    --dataset_file data/spider_role/dev_role.json \
    --database_path spider/database \
    --output_file results/spider_eval.json
```

### Evaluating on BIRD

```bash
python experiments/eval/evaluation_bird_role.py \
    --model_path outputs/qwen2.5_bird_role/checkpoint-best \
    --dataset_file data/bird_role/dev_role.json \
    --output_file results/bird_eval.json
```

### Evaluating on LiveSQLBench

```bash
python experiments/eval/evaluation_livesqlbench_role.py \
    --model_path outputs/qwen2.5_livesql_role/checkpoint-best \
    --dataset_file data/livesqlbench_role/questions_role.json \
    --output_file results/livesql_eval.json
```

### Evaluation Metrics

The evaluation script computes:
- **EX (Execution Accuracy)**: % of queries producing correct results
- **Answerable Rate**: % of answerable queries the model attempts
- **Correct Refusal Rate**: % of unanswerable queries correctly refused
- **Incorrect Refusal Rate**: % of answerable queries incorrectly refused
- **Violation Rate**: % of unanswerable queries where model generates SQL

---



## Fine-tuning (Optional)

If you want to fine-tune your own models instead of using cloud APIs:

### Downloading Pre-trained Models

Download models from Hugging Face:

```bash
# Qwen2.5-14B (recommended)
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir models/qwen2.5-14b

# Llama-3-8B (alternative)
huggingface-cli download meta-llama/Llama-3-8B-Instruct --local-dir models/llama3-8b
```

### Training a Role-Aware Model

Train using QLoRA on Spider role-aware dataset:

```bash
cd experiments/train
bash train_sft.sh
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
CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_sft.sh
```

Training uses:
- **4-bit quantization** (QLoRA) for memory efficiency
- **LoRA adapters** (rank=64) for parameter-efficient fine-tuning
- **DeepSpeed ZeRO-3** for distributed training
- **Causal LM objective** with assistant-only loss (prompt masked with `IGNORE_INDEX=-100`)

---



## Code Structure

```
Role-SQL-benchmark/
├── config/
│   └── eval_config.yaml              # Evaluation configuration
├── configs/
│   ├── prompts.py                    # System/user prompts for role generation
│   └── role_templates.yaml           # Predefined role templates
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
├── scripts/
│   ├── init_all.sh                   # Initialize all PostgreSQL databases
│   ├── check_pg_connection.sh        # Verify database connectivity
│   └── pg_env.sh                     # PostgreSQL environment variables
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

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{role-sql-2024,
  title={Role-Based Access Control for Text-to-SQL: Benchmark and Evaluation},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

## License

This project is licensed under the Apache License 2.0. Third-party datasets (Spider, BIRD) retain their original licenses.
