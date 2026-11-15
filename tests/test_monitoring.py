"""Contrato temporal congelado y calidad de monitoreo P1."""
import json

import numpy as np
import pandas as pd
import pytest

from mibici.dataset import DataError
from mibici.features import full_weeks, p1_station_week_panel
from mibici.monitoring import COLUMNS, current_panel, frame_digest, load_partitions, validate_frame

MONTHS = pd.period_range("2024-10", "2026-09", freq="M").astype(str).tolist()


def data():
    # Incluye la parte de septiembre de la semana que cruza a octubre.
    weeks = pd.date_range("2024-10-07", "2026-09-28", freq="7D")
    old = pd.DataFrame({"station_id": 1, "week_start": weeks, "departures": 10})
    new = pd.DataFrame({"station_id": [1, 1, 1, 1, 1, 2],
                        "week_start": pd.to_datetime(["2026-09-28", "2026-10-05", "2026-10-12",
                                                      "2026-10-19", "2026-10-26", "2026-10-05"]),
                        "departures": [20, 40, 50, 60, 70, 999]})
    panel, _ = p1_station_week_panel(old, [1], MONTHS)
    return old, new, panel


def test_current_freezes_cohort_and_adds_cross_month_history():
    old, new, panel = data()
    saved = panel.copy(deep=True)
    current = current_panel(old, new, panel, MONTHS, ["2026-10"])
    pd.testing.assert_frame_equal(saved, panel)
    assert set(current.station_id) == {1}  # estación futura excluida
    assert current.week_start.tolist() == list(full_weeks(["2026-10"]))
    assert current.lag_1.tolist() == [30., 40., 50.]
    assert current.iloc[0].mean_previous_4 == 15.
    assert current.iloc[0].lag_52 == 10.
    assert current.week_start.max() == pd.Timestamp("2026-10-19")
    assert list(current.columns) == COLUMNS


def test_current_target_cannot_change_its_own_features():
    old, new, panel = data()
    before = current_panel(old, new, panel, MONTHS, ["2026-10"])
    new.loc[new.week_start >= "2026-10-12", "departures"] += 10000
    after = current_panel(old, new, panel, MONTHS, ["2026-10"])
    pd.testing.assert_series_equal(before.iloc[1][COLUMNS[3:]], after.iloc[1][COLUMNS[3:]])
    assert after.iloc[1].departures != before.iloc[1].departures


def test_gap_in_months_rejected():
    old, new, panel = data()
    with pytest.raises(DataError, match="consecutivos"):
        current_panel(old, new, panel, MONTHS, ["2026-11"])


@pytest.mark.parametrize("column,value", [("lag_1", np.nan), ("lag_52", np.inf),
                                        ("departures", -1), ("week_cos", 100)])
def test_invalid_monitoring_features_rejected(column, value):
    old, new, panel = data()
    frame = current_panel(old, new, panel, MONTHS, ["2026-10"])
    frame.loc[0, column] = value
    with pytest.raises(DataError):
        validate_frame(frame)


def test_lag52_missing_is_valid_and_hash_is_portable():
    old, new, panel = data()
    frame = current_panel(old, new, panel, MONTHS, ["2026-10"])
    frame.loc[0, "lag_52"] = np.nan
    validate_frame(frame)
    changed = frame.copy()
    changed.loc[0, "week_sin"] += 1e-15
    assert frame_digest(frame) == frame_digest(changed)
    changed.loc[0, "lag_1"] += 1
    assert frame_digest(frame) != frame_digest(changed)


def test_tampered_metadata_rejected(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"contract_id": "incorrect", "partitions": {}}), encoding="utf-8")
    with pytest.raises(DataError, match="Metadatos"):
        load_partitions(tmp_path)
