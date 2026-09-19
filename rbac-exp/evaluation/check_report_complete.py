"""Release gate: refuse to mix legacy, partial, or differently sampled reports."""
import argparse
import json
from pathlib import Path


def check(reports, require_ex=False):
    first = None
    schemas_seen = {}
    for report in reports:
        if report.get("protocol") != "rbac-six-category-v2":
            raise ValueError("legacy or missing protocol: rerun using protocol v2")
        signature = (report["dataset_sha256"], report["sqlglot_version"], report["schema_provenance"],
                     [(t["seed"], t["selected_source_indices"]) for t in report["trials"]])
        # Each method may touch different DBs; compare common schema hashes.
        if first is not None:
            if signature[:2] != first[:2] or signature[3] != first[3]:
                raise ValueError("methods differ in dataset, parser, or sampled source indices")
            for db in set(schemas_seen) & set(signature[2]):
                if schemas_seen[db] != signature[2][db]:
                    raise ValueError("schema mismatch for " + db)
        if first is None:
            first = signature
        schemas_seen.update(signature[2])
        for trial in report["trials"]:
            row = trial["overall"]
            if row["policy_unknown_count"] or row["ac_f1"] is None:
                raise ValueError("unresolved ALLOW policy checks: AC-F1 not ready for release")
            if row["sql_policy_unknown_count"]:
                raise ValueError("unresolved SQL checks: actual violation rate not ready for release")
            if require_ex and row["execution_unknown_count"]:
                raise ValueError("missing per-row execution judgments: six categories not complete")
    if not reports:
        raise ValueError("no reports")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", nargs="+")
    p.add_argument("--require_ex", action="store_true")
    args = p.parse_args()
    check([json.loads(Path(x).read_text(encoding="utf-8")) for x in args.reports], args.require_ex)
    print("Protocol, coverage, dataset, schema and paired sampling checks passed.")


if __name__ == "__main__":
    main()
