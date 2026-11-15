"""Prueba real opcional: ejecutar en requirements-monitoring.txt."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("evidently")

from mibici.monitoring import FEATURES
from mibici.monitoring_report import evaluate


def test_real_evidently_detects_shift_and_reports_missing_values(tmp_path):
    rng = np.random.default_rng(13)
    reference = pd.DataFrame(rng.normal(size=(150, 5)), columns=FEATURES)
    reference.loc[:29, "lag_52"] = np.nan
    current = reference.copy()
    current["lag_1"] += 20
    report = evaluate(reference, current)
    records = report.dict()["metrics"]
    drift = {r["config"].get("column"): r for r in records
             if r["config"]["type"].endswith(":ValueDrift")}
    assert set(drift) == set(FEATURES)
    assert drift["lag_1"]["value"] > .1
    assert drift["week_sin"]["value"] == 0
    missing = [r for r in records if r["config"]["type"].endswith(":MissingValueCount")
               and r["config"].get("column") == "lag_52"]
    assert missing[0]["value"]["count"] == 30
    report.save_html(str(tmp_path / "evidently.html"))
    assert (tmp_path / "evidently.html").stat().st_size > 1000
