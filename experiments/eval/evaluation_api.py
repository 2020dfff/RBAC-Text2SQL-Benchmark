from typing import Any, Dict, Optional

from experiments.eval import evaluation


def start_evaluate(
    args: Optional[Dict[str, Any]] = None,
):
    # Arguments for evaluation
    if args is None:
        args = {
            "input": "./experiments/output/pred/pred_sql_dev_skeleton.sql",
            "gold": "./data/spider/dev_gold.sql",
            "gold_natsql": "./data/spider/dev_gold.sql",
            "db": "./data/spider/database",
            "table": "./data/spider/tables.json",
            "table_natsql": "./data/spider/tables.json",
            "etype": "exec",
            "plug_value": True,
            "keep_distict": False,
            "progress_bar_for_each_datapoint": False,
            "natsql": False,
            "difficulty_json": "./data/spider/dev.json",
        }
    else:
        args = args

    # Execute evaluation
    evaluation.evaluate_api(args)


if __name__ == "__main__":
    start_evaluate()
