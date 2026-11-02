"""Solo contrato y suites de calidad; no entrena ni descarga datos."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from mibici.validation import demo_panel, panel_errors, run_checks
from mibici.validation_tasks import frame_errors, synthetic_frame, run_question


class AllQuestionsContractTests(unittest.TestCase):
    def test_p1_and_p3_valid_contracts(self):
        for question in ["P1", "P3"]:
            with self.subTest(question=question):
                self.assertEqual(frame_errors(synthetic_frame(question), question), [])

    def test_p1_rejects_overlapping_target_weeks(self):
        frame = synthetic_frame("P1")
        frame.loc[frame["split"].eq("test"), "week_start"] = "2025-01-06"
        self.assertTrue(frame_errors(frame, "P1"))

    def test_p1_rejects_negative_demand(self):
        frame = synthetic_frame("P1")
        frame.loc[0, "target"] = -1
        self.assertTrue(frame_errors(frame, "P1"))

    def test_p3_rejects_inconsistent_aggregates_and_extra_columns(self):
        frame = synthetic_frame("P3")
        frame.loc[frame.index[0], "traffic"] += 1
        self.assertTrue(frame_errors(frame, "P3"))
        frame = synthetic_frame("P3")
        frame["user_id"] = 13
        self.assertTrue(frame_errors(frame, "P3"))

    def test_p3_preserves_zero_capacity_and_signed_balance(self):
        frame = synthetic_frame("P3")
        frame["capacity"] = 0
        frame["departures"], frame["arrivals"] = 0, 10
        frame["traffic"], frame["balance"] = 10, -10
        self.assertEqual(frame_errors(frame, "P3"), [])


class ValidationContractTests(unittest.TestCase):
    def test_valid_panel(self):
        self.assertEqual(panel_errors(demo_panel()), [])

    def test_missing_column(self):
        self.assertTrue(panel_errors(demo_panel().drop(columns="capacity")))

    def test_unexpected_column_never_reaches_report(self):
        panel = demo_panel()
        panel["usuario_id"] = "not-for-report"
        self.assertTrue(panel_errors(panel))

    def test_numeric_type_is_not_silently_coerced(self):
        panel = demo_panel()
        panel["capacity"] = panel.capacity.astype(str)
        self.assertTrue(panel_errors(panel))

    def test_invalid_target_and_infinity(self):
        panel = demo_panel()
        panel.loc[0, "target"] = 2
        self.assertTrue(panel_errors(panel))
        panel = demo_panel()
        panel.loc[0, "net_flow_mean4"] = np.inf
        self.assertTrue(panel_errors(panel))

    def test_temporal_overlap_and_horizon(self):
        panel = demo_panel()
        panel.loc[panel["split"] == "test", "target_week"] = "2025-01-13"
        self.assertTrue(panel_errors(panel))

    def test_numeric_null_is_retained_for_deepchecks(self):
        panel = demo_panel()
        panel.loc[0, "capacity"] = np.nan
        self.assertEqual(panel_errors(panel), [])

    def test_schema_failure_is_nonzero_and_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _ = run_checks(demo_panel().drop(columns="target"), tmp,
                                 {"kind": "synthetic_test"})
            self.assertEqual(code, 1)
            result = json.loads((Path(tmp) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "SCHEMA_FAILED")


@unittest.skipUnless(importlib.util.find_spec("deepchecks"), "Falta el entorno Deepchecks")
class DeepchecksIntegrationTests(unittest.TestCase):
    def test_p1_real_suites_allow_expected_seasonal_missingness(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, summary = run_question(synthetic_frame("P1"), "P1", tmp, {"kind": "synthetic_test"})
            self.assertEqual(code, 0)
            self.assertTrue(all(r["passed"] for r in summary["integrity"].values()))
            self.assertEqual(summary["allowed_missing_features"], ["lag_52"])
            self.assertEqual(len(list(Path(tmp).glob("*.html"))), 5)

    def test_p1_unexpected_null_is_rejected(self):
        frame = synthetic_frame("P1")
        frame.loc[0, "lag_1"] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            code, summary = run_question(frame, "P1", tmp, {"kind": "synthetic_test"})
            self.assertEqual(code, 1)
            self.assertFalse(summary["integrity"]["train"]["passed"])

    def test_p3_has_no_fabricated_drift_or_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, summary = run_question(synthetic_frame("P3"), "P3", tmp, {"kind": "synthetic_test"})
            self.assertEqual(code, 0)
            self.assertTrue(summary["integrity"]["full"]["passed"])
            self.assertEqual(summary["drift"], {})
            self.assertIn("drift_not_applicable", summary)
            self.assertEqual(len(list(Path(tmp).glob("*.html"))), 1)

    def test_p3_duplicate_station_is_rejected(self):
        frame = synthetic_frame("P3")
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as tmp:
            code, summary = run_question(frame, "P3", tmp, {"kind": "synthetic_test"})
            self.assertEqual(code, 1)
            self.assertFalse(summary["integrity"]["full"]["passed"])

    def test_demo_runs_all_checks_and_creates_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, summary = run_checks(demo_panel(), tmp, {"kind": "synthetic_test"})
            self.assertEqual(code, 0)
            self.assertTrue(all(r["passed"] for r in summary["integrity"].values()))
            self.assertTrue(all(r["checks_not_run"] == 0 for r in summary["drift"].values()))
            self.assertEqual(len(list(Path(tmp).glob("*.html"))), 5)

    def test_duplicate_station_week_is_rejected_by_deepchecks(self):
        panel = demo_panel()
        panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as tmp:
            code, summary = run_checks(panel, tmp, {"kind": "synthetic_test"})
            self.assertEqual(code, 1)
            self.assertFalse(summary["integrity"]["train"]["passed"])


if __name__ == "__main__":
    unittest.main()
