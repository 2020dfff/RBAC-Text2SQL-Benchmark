"""Regression checks for the v2 correction and permission-check boundaries."""
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rbac-exp"))
from evaluation.access_protocol import (classify, make_case, score_cases, select_trials,
                                        validate_execution_cache)
from evaluation.prediction_policy import PolicyChecker
from evaluation.evaluate_protocol_v2 import evaluate
from evaluation.check_report_complete import check as check_release


class ProtocolTests(unittest.TestCase):
    def case(self, allowed, refusal=False, status="PASS", ex=True):
        row = {"allowed": allowed, "gold_sql": "SELECT x FROM t"}
        return make_case(0, row, "SELECT x FROM t", refusal, {"status": status}, ex)

    def test_six_categories_and_only_new_transition(self):
        for allowed in (False, True):
            self.assertEqual(classify(allowed, True, "NOT_APPLICABLE", None),
                             "incorrect_refusal" if allowed else "correct_refusal")
            for ex in (False, True):
                self.assertEqual(classify(allowed, False, "BLOCK", ex),
                                 "violation_correct" if ex else "violation_wrong")
                self.assertEqual(classify(allowed, False, "PASS", ex),
                                 ("correct" if ex else "wrong") if allowed else
                                 ("violation_correct" if ex else "violation_wrong"))

    def test_rejection_does_not_remove_a_true_positive(self):
        common = [self.case(True), self.case(False, True)]
        before = score_cases(common + [self.case(True, status="BLOCK")])
        after = score_cases(common + [self.case(True, True)])
        self.assertEqual(before["ac_f1"], after["ac_f1"])
        self.assertGreater(before["legacy_decision_ac_f1"], after["legacy_decision_ac_f1"])

    def test_safe_ex_fixed_denominator_and_cached_ex_conservation(self):
        cases = [self.case(True), self.case(True, status="BLOCK"), self.case(False, status="PASS")]
        result = score_cases(cases)
        self.assertEqual(result["safe_ex_denominator"], 2)
        self.assertEqual(result["safe_ex"], .5)
        self.assertEqual(result["counts"]["correct"] + result["counts"]["violation_correct"], 3)
        self.assertEqual(result["sql_policy_violation_count"], 1)
        self.assertEqual(result["failure_to_refuse_count"], 1)

    def test_unknown_is_not_pass_and_missing_ex_is_not_wrong(self):
        result = score_cases([self.case(True, status="UNKNOWN"), self.case(False, status="UNKNOWN")])
        self.assertIsNone(result["ac_f1"])
        self.assertEqual(result["ac_f1_lower"], 0)
        self.assertAlmostEqual(result["ac_f1_upper"], 2 / 3)
        self.assertEqual(result["counts"]["violation_correct"], 1)
        result = score_cases([self.case(True, ex=None)])
        self.assertEqual(result["ac_f1"], 1)
        self.assertEqual(result["counts"]["wrong"], 0)
        self.assertIsNone(result["safe_ex"])

    def test_alignment_hashes(self):
        row = {"allowed": True, "gold_sql": "SELECT x FROM t"}
        case = self.case(True)
        self.assertEqual(validate_execution_cache([case], [row], [case["prediction"]]), {0: True})
        with self.assertRaises(ValueError):
            validate_execution_cache([case], [row], ["SELECT y FROM t"])
        with self.assertRaises(ValueError):
            validate_execution_cache([case, case], [row], [case["prediction"]])

    def test_sampling_keeps_source_indices(self):
        rows = [{"input": "a"}, {"input": "b"}, {"input": "a"}]
        trials = select_trials(rows, True, 5)
        self.assertEqual([t["seed"] for t in trials], [42, 43, 44, 45, 46])
        for t in trials:
            self.assertEqual(len(t["selected_source_indices"]), 2)
            self.assertIn(1, t["selected_source_indices"])

    def test_release_gate_rejects_legacy_unknown_and_misaligned_trials(self):
        from copy import deepcopy
        result = score_cases([self.case(True)])
        report = {"protocol": "rbac-six-category-v2", "dataset_sha256": "dataset", "sqlglot_version": "30.18.0",
                  "schema_provenance": {}, "trials": [{"seed": 42, "selected_source_indices": [2], "overall": result}]}
        check_release([report, report], require_ex=True)
        bad = deepcopy(report); bad["protocol"] = "legacy"
        with self.assertRaises(ValueError):
            check_release([bad])
        bad = deepcopy(report); bad["trials"][0]["selected_source_indices"] = [3]
        with self.assertRaises(ValueError):
            check_release([report, bad])
        bad = deepcopy(report); bad["trials"][0]["overall"] = score_cases([self.case(True, status="UNKNOWN")])
        with self.assertRaises(ValueError):
            check_release([bad])


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.schema = {"db": {"tables": {"t": ["id", "x", "secret"], "u": ["id", "y"]}}}
        self.checker = PolicyChecker(schemas=self.schema)
        self.policy = {"t": ["id", "x"], "u": ["id", "y"]}

    def check(self, sql, policy=None):
        return self.checker.check("db", sql, self.policy if policy is None else policy)

    def test_hidden_predicate_is_blocked(self):
        r = self.check("SELECT x FROM t WHERE secret > 1")
        self.assertEqual(r["status"], "BLOCK", r)
        self.assertIn({"operation": "SELECT", "table": "t", "column": "secret"}, r["missing"])

    def test_table_access_count_star_and_star_expansion(self):
        self.assertEqual(self.check("SELECT COUNT(*) FROM t", {})["status"], "BLOCK")
        self.assertEqual(self.check("SELECT COUNT(*) FROM t", {"t": []})["status"], "PASS")
        self.assertEqual(self.check("SELECT * FROM t")["status"], "BLOCK")
        self.assertEqual(self.check("SELECT COUNT(x) FROM t", {"t": []})["status"], "BLOCK")

    def test_scopes_and_correlated_subquery(self):
        for sql in (
            "WITH c AS (SELECT x FROM t) SELECT x FROM c",
            "SELECT a.x FROM t a WHERE EXISTS (SELECT 1 FROM u b WHERE b.id=a.id)",
            "SELECT x FROM t UNION SELECT y FROM u",
            "SELECT a.x FROM t a JOIN u b USING(id)",
        ):
            with self.subTest(sql=sql):
                self.assertEqual(self.check(sql)["status"], "PASS", self.check(sql))
        self.assertEqual(self.check("WITH c AS (SELECT secret FROM t) SELECT * FROM c")["status"], "BLOCK")

    def test_uncertain_sql_is_not_silently_authorized(self):
        for sql in ("SELECT missing FROM t", "SELECT id FROM t JOIN u ON t.id=u.id",
                    "SELECT x FROM nonexistent", "garbage", "", "SELECT * FROM t NATURAL JOIN u"):
            with self.subTest(sql=sql):
                self.assertEqual(self.check(sql)["status"], "UNKNOWN", self.check(sql))

    def test_all_statements_checked(self):
        self.assertEqual(self.check("SELECT x FROM t; SELECT secret FROM t")["status"], "BLOCK")

    def test_sqlite_schema_is_read_without_query_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "db"
            path.mkdir()
            with contextlib.closing(sqlite3.connect(path / "db.sqlite")) as con:
                con.execute("CREATE TABLE t(id INT, x INT, secret INT)")
            checker = PolicyChecker(db_root=temp)
            self.assertEqual(checker.check("db", "SELECT secret FROM t", self.policy)["status"], "BLOCK")
            self.assertEqual(checker.check("db", "DROP TABLE t", self.policy)["status"], "UNKNOWN")
            with contextlib.closing(sqlite3.connect(path / "db.sqlite")) as con:
                self.assertEqual(con.execute("SELECT COUNT(*) FROM t").fetchone(), (0,))

    def test_crud_target_and_read_dependencies(self):
        checker = PolicyChecker(schemas=self.schema, dialect="postgres")
        policy = {"tables": {"t": {"SELECT": ["id", "x"], "UPDATE": ["x"]}}, "INSERT": ["t"], "DELETE": ["t"], "DDL": False}
        for sql, expected in (
            ("UPDATE t SET x=1 WHERE id=2", "PASS"),
            ("UPDATE t SET x=secret WHERE id=2", "BLOCK"),
            ("UPDATE t SET secret=1", "BLOCK"),
            ("DELETE FROM t WHERE secret=1", "BLOCK"),
            ("INSERT INTO t(x) VALUES (1)", "PASS"),
            ("INSERT INTO t(x) SELECT secret FROM t", "BLOCK"),
            ("DO $$ BEGIN NULL; END $$", "UNKNOWN"),
        ):
            with self.subTest(sql=sql):
                r = checker.check("db", sql, policy, crud=True)
                self.assertEqual(r["status"], expected, r)

    def test_offline_end_to_end_and_length_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [{"allowed": True, "db_id": "db", "policy": self.policy, "gold_sql": "SELECT x FROM t"}]
            (root / "rows.json").write_text(json.dumps(rows))
            (root / "pred.sql").write_text("SELECT secret FROM t\n")
            with contextlib.redirect_stdout(io.StringIO()):
                r = evaluate(root / "rows.json", root / "pred.sql", schemas=self.schema, output_dir=root / "out")
            self.assertEqual(r["trials"][0]["overall"]["ac_f1"], 0)
            self.assertEqual(r["trials"][0]["overall"]["counts"]["violation_unscored"], 1)
            (root / "pred.sql").write_text("SELECT x FROM t\nSELECT x FROM t\n")
            with self.assertRaises(ValueError):
                evaluate(root / "rows.json", root / "pred.sql", schemas=self.schema, output_dir=root / "out")


if __name__ == "__main__":
    unittest.main()
