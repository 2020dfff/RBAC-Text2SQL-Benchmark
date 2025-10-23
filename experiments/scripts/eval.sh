# ==================================== For Spider evaluation =====================================
# ==================================== without role =====================================
# python experiments/eval/evaluation_spider.py \
#     --input experiments/output/pred/pred_qwen2.5-coder-7b-bird-role_cleaned.sql \
#     --difficulty_json data/selected/spider/spider_dev.json \
#     --etype exec \
#     --fair_comparison

# ==================================== with role =====================================
# python experiments/eval/evaluation_spider_role.py \
#     --input experiments/output/pred/pred_gemini-2.5-flash_spider_role.sql \
#     --role_json data/selected/spider/spider_dev_with_role.json \
#     --etype exec \
#     --fair_comparison

# ==================================== For Bird evaluation =====================================
# ==================================== without role =====================================
# python experiments/eval/evaluation_bird.py \
#      --predicted_sql_path "experiments/output/pred/pred_xxx_bird_cleaned.sql" \
#      --ground_truth_path "data/selected/gold_data/bird_dev_gold.txt" \
#      --db_root_path "data/Bird/dev_20240627/dev_databases/" \
#      --diff_json_path "data/selected/gold_data/bird_difficulty.txt" \
#      --etype exec \
#      --num_cpus 4 \
#      --meta_time_out 20

# ==================================== with role =====================================
# python experiments/eval/evaluation_bird_role.py \
#      --input "experiments/output/pred/pred_qwen2.5-coder-7b-bird-role_cleaned.sql" \
#      --gold "data/selected/gold_data/bird_gold_role.txt" \
#      --role_json "data/selected/bird/bird_dev_with_role.json" \
#      --db_root_path "data/Bird/dev_20240627/dev_databases/" \
#      --etype exec \
#      --num_cpus 4 \
    # --fair_comparison

# ==================================== with role =====================================
# python experiments/eval/evaluation_bird_role.py \
#     --input "./experiments/output/pred/pred_claude-sonnet-4-5_bird_role.sql" \
#     --gold "./dbgpt_hub_sql/data/eval_data/bird_gold_role.txt" \
#     --role_json "./data/selected/bird/bird_dev_with_role.json" \
#     --db_root_path "./data/selected/bird/dev/dev_databases/" \
#     --etype exec \
#     --num_cpus 40 \
#     --fair_comparison

# ========================= with role and rbac settings ==============================

# python experiments/eval/evaluation_bird_role.py \
#     --input "./experiments/output/pred/pred_google-gemma-3-27b-it_bird_rbac.sql" \
#     --gold "./dbgpt_hub_sql/data/eval_data/bird_gold_role.txt" \
#     --role_json "./data/selected/bird/bird_dev_with_role_rbac.json" \
#     --db_root_path "./data/selected/bird/dev/dev_databases/" \
#     --etype exec \
#     --num_cpus 40 \
#     --fair_comparison

# ==================================== For livesqlbench evaluation =====================================
# ==================================== without role =====================================
# PYTHONPATH=. python experiments/scripts/convert_livesqlbench_predictions.py \
#     --pred-sql experiments/output/pred/pred_google-gemma-3-27b-it_livesqlbench.sql \
#     --dataset data/selected/livesqlbench/livesqlbench_dev.json \
#     --output experiments/output/pred/pred_google-gemma-3-27b-it_livesqlbench.jsonl

# PYTHONPATH=. python experiments/eval/evaluation_livesqlbench.py \
#     --predictions experiments/output/pred/pred_google-gemma-3-27b-it_livesqlbench.jsonl \
#     --base-data data/livesqlbench-base-lite-sqlite/livesqlbench_data_sqlite.jsonl \
#     --gold-sql data/livesqlbench-base-lite-sqlite/livesqlbench_sqlite_gt_kg_testcases_0528.jsonl \
#     --db-root data/livesqlbench-base-lite-sqlite \
#     --output-dir experiments/output/eval_result

# ==================================== with role =====================================
# PYTHONPATH=. python experiments/scripts/convert_livesqlbench_predictions.py \
#   --pred-sql experiments/output/pred/pred_claude-sonnet-4-5_livesqlbench_role.sql \
#   --dataset data/selected/livesqlbench/livesqlbench_dev_with_role.json \
#   --output experiments/output/pred/pred_claude-sonnet-4-5_livesqlbench_role.jsonl

# PYTHONPATH=. python experiments/eval/evaluation_livesqlbench_role.py \
#   --predictions experiments/output/pred/pred_claude-sonnet-4-5_livesqlbench_role.jsonl \
#   --role-dataset data/selected/livesqlbench/livesqlbench_dev_with_role.json \
#   --base-data data/selected/livesqlbench/dev/livesqlbench_data_sqlite.jsonl \
#   --gold-sql data/selected/livesqlbench/dev/livesqlbench_sqlite_gt_kg_testcases_0528.jsonl \
#   --db-root data/selected/livesqlbench/dev \
#   --output-dir experiments/output/eval_result \
#   --fair-comparison
