"""Pruebas de geografía, cortes y disponibilidad temporal con registros sintéticos."""

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, mapping

from mibici import config
from mibici.dataset import DataError
from mibici.features import full_weeks, p1_station_week_panel, weekly_departures
from mibici.geography import boundary_geometry, municipality_station_ids
from mibici.modeling.p1 import baseline_predictions, regression_metrics

MONTHS = pd.period_range("2025-10", "2026-09", freq="M").astype(str).tolist()


def counts():
    weeks = full_weeks(MONTHS)
    return pd.DataFrame({"station_id": 1, "week_start": weeks,
                         "departures": np.arange(1, len(weeks) + 1)})


def test_only_complete_weeks_in_source_interval():
    weeks = full_weeks(MONTHS)
    assert weeks[0] == pd.Timestamp("2025-10-06")
    assert weeks[-1] == pd.Timestamp("2026-09-21")
    assert (weeks.dayofweek == 0).all()
    with pytest.raises(DataError):
        full_weeks(["2025-10", "2025-12"])


def test_common_temporal_split_and_lags():
    panel, metadata = p1_station_week_panel(counts(), [1], MONTHS)
    assert metadata["weeks_by_split"] == {"train": 31, "validation": 8, "test": 8}
    assert metadata["validation_start"] == "2026-06-08"
    assert metadata["test_start"] == "2026-08-03"
    assert metadata["test_end_exclusive"] == "2026-09-28"
    assert panel.iloc[0].departures == 5
    assert panel.iloc[0].lag_1 == 4
    assert panel.iloc[0].mean_previous_4 == 2.5
    assert panel[panel.split == "train"].week_start.max() < panel[panel.split == "validation"].week_start.min()
    assert panel[panel.split == "validation"].week_start.max() < panel[panel.split == "test"].week_start.min()


def test_future_target_cannot_change_its_own_features():
    original = counts()
    before, _ = p1_station_week_panel(original, [1], MONTHS)
    target = pd.Timestamp("2026-08-03")
    changed = original.copy()
    changed.loc[changed.week_start >= target, "departures"] += 10000
    after, _ = p1_station_week_panel(changed, [1], MONTHS)
    for column in ("lag_1", "mean_previous_4"):
        assert before.loc[before.week_start == target, column].iloc[0] == after.loc[after.week_start == target, column].iloc[0]
    assert after.loc[after.week_start == target, "departures"].iloc[0] != before.loc[before.week_start == target, "departures"].iloc[0]


def test_future_only_station_is_not_added_to_training_cohort():
    original = counts()
    new = pd.DataFrame({"station_id": [2], "week_start": [pd.Timestamp("2026-08-03")], "departures": [20]})
    panel, meta = p1_station_week_panel(pd.concat([original, new]), [1, 2], MONTHS)
    assert set(panel.station_id) == {1}
    assert meta["new_station_ids_excluded"] == [2]


def test_no_zero_rows_before_first_training_observation():
    frame = counts().iloc[10:].copy()
    # Ausencia de una semana intermedia sí representa cero observado.
    missing = frame.iloc[6].week_start
    frame = frame[frame.week_start != missing]
    panel, _ = p1_station_week_panel(frame, [1], MONTHS)
    assert panel.week_start.min() == full_weeks(MONTHS)[14]
    assert panel.loc[panel.week_start == missing, "departures"].iloc[0] == 0


def test_short_period_rejected():
    with pytest.raises(DataError, match="suficientes"):
        p1_station_week_panel(counts(), [1], ["2026-09"])


def test_geography_covers_interior_and_boundary_not_names():
    geometry = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    stations = pd.DataFrame({"station_id": [1, 2, 3], "longitude": [1, 0, 3],
                             "latitude": [1, 1, 1],
                             "station_name": ["Sin prefijo", "ex-otra", "ZPN-fuera"]})
    assert municipality_station_ids(stations, geometry) == [1, 2]


def test_boundary_identity_verified():
    feature = {"type": "Feature", "properties": {"cvegeo": config.P1_MUNICIPALITY_CODE,
                "nomgeo": config.P1_MUNICIPALITY_NAME},
               "geometry": mapping(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))}
    doc = {"type": "FeatureCollection", "features": [feature]}
    assert boundary_geometry(doc).is_valid
    feature["properties"]["nomgeo"] = "Otro municipio"
    with pytest.raises(DataError):
        boundary_geometry(doc)


def test_predictions_and_metrics():
    panel, _ = p1_station_week_panel(counts(), [1], MONTHS)
    first = baseline_predictions(panel, "previous_week")
    second = baseline_predictions(panel, "mean_previous_weeks")
    assert np.all(first < panel.departures)
    assert np.all(second < first)
    assert regression_metrics([0, 10], [2, 8]) == {"mae": 2.0, "rmse": 2.0, "wape": 0.4}
    assert regression_metrics([0, 0], [0, 0]) == {"mae": 0.0, "rmse": 0.0}
    with pytest.raises(DataError):
        regression_metrics([1], [np.nan])


def test_weekly_aggregation_reads_only_operational_fields(tmp_path):
    path = tmp_path / "trips.parquet"
    pd.DataFrame({"origin_station_id": [1, 1, 2],
                  "started_at": pd.to_datetime(["2026-09-06 23:59", "2026-09-07 00:00", "2026-09-07 01:00"]),
                  "ignored_field": ["synthetic", "synthetic", "synthetic"]}).to_parquet(path)
    manifest = {"files": [{"kind": "trips", "path": "trips.parquet"}]}
    result = weekly_departures(manifest, tmp_path)
    assert result.departures.sum() == 3
    assert result.columns.tolist() == ["station_id", "week_start", "departures"]
    assert set(result.week_start) == {pd.Timestamp("2026-08-31"), pd.Timestamp("2026-09-07")}


def test_two_years_keep_recent_cohort_and_same_validation_test_rows():
    long_months = pd.period_range("2024-10", "2026-09", freq="M").astype(str).tolist()
    long_weeks = full_weeks(long_months)
    history = pd.DataFrame({"station_id": 1, "week_start": long_weeks,
                            "departures": np.arange(1, len(long_weeks) + 1)})
    old_only = history[history.week_start < "2025-10-01"].copy()
    old_only["station_id"] = 2
    combined = pd.concat([history, old_only], ignore_index=True)
    short = history[history.week_start >= "2025-10-01"]
    yearly, _ = p1_station_week_panel(short, [1, 2], MONTHS)
    extended, metadata = p1_station_week_panel(combined, [1, 2], long_months,
                                              cohort_start_month="2025-10")
    assert set(extended.station_id) == {1}
    assert metadata["weeks_by_split"]["train"] > 31
    pd.testing.assert_frame_equal(
        yearly[yearly.split != "train"][["station_id", "week_start", "lag_1", "mean_previous_4"]].reset_index(drop=True),
        extended[extended.split != "train"][["station_id", "week_start", "lag_1", "mean_previous_4"]].reset_index(drop=True))


def test_seasonal_baseline_uses_past_or_explicit_fallback():
    panel = pd.DataFrame({"lag_1": [10, 20], "lag_52": [5, np.nan]})
    assert baseline_predictions(panel, "seasonal_52_weeks").tolist() == [5, 20]
