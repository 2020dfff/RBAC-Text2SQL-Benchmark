"""Version routing for the historical evaluator entrypoints."""
import sys
from pathlib import Path
import warnings


def dispatch_cli(crud=False):
    """Return only when explicit legacy reproduction was requested."""
    if "--legacy_protocol_v1" in sys.argv:
        sys.argv.remove("--legacy_protocol_v1")
        warnings.warn("LEGACY v1: predicted SQL policy is NOT checked; do not report as v2.", RuntimeWarning)
        return
    from evaluation.evaluate_protocol_v2 import main
    main(crud=crud)
    raise SystemExit(0)


def read_execution_callback(db_dir, plug_value=False):
    import sqlglot
    from sqlglot import exp
    from evaluation.evaluate_column_level import exec_match_with_timeout
    def execute(row, prediction, gold):
        for text in (prediction, gold):
            try:
                trees = [x for x in sqlglot.parse(text, read="sqlite") if x is not None]
                if len(trees) != 1 or not isinstance(trees[0], exp.Query):
                    return False
                if any(isinstance(n, (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Command)) for n in trees[0].walk()):
                    return False
                if any(n.args.get("into") or n.args.get("locks") for n in trees[0].find_all(exp.Select)):
                    return False
            except sqlglot.errors.ParseError:
                return False
        path = (Path(db_dir) / row["db_id"] / (row["db_id"] + ".sqlite")).resolve()
        if not path.is_relative_to(Path(db_dir).resolve()) or not path.is_file():
            raise ValueError("missing or invalid execution database path")
        return exec_match_with_timeout(str(path), prediction, gold, plug_value=plug_value)
    return execute


def evaluate_read_api(role_json_file, predict_file, dataset, db_dir=None, output_dir=None,
                      execute_sql=True, fair_comparison=False, plug_value=False, num_trials=5,
                      execution_cache=None, schemas=None):
    from evaluation.evaluate_protocol_v2 import evaluate
    if db_dir is None:
        from configs.paths import get_dataset_paths
        db_dir = get_dataset_paths(dataset)["db"]
    return evaluate(role_json_file, predict_file, db_dir=db_dir, output_dir=output_dir,
                    fair_comparison=fair_comparison, num_trials=num_trials,
                    execution_cache=execution_cache, schemas=schemas,
                    execute=read_execution_callback(db_dir, plug_value) if execute_sql else None)
