"""Reproducibilidad del entrenamiento P1 con datos sintéticos, sin red ni MLflow."""

import json

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from mibici import config
from mibici.features import full_weeks, p1_station_week_panel
from mibici.modeling.p1 import regression_metrics
from mibici.modeling.p1_candidates import candidate_features, candidate_specs


@pytest.mark.parametrize("missing_prediction_history", [False, True],
                         ids=["partial_annual_history", "missing_annual_history_at_prediction"])
def test_p1_training_reproduces_predictions_and_metrics(missing_prediction_history):
    """Dos estimadores nuevos ajustan solo train y coinciden en validation/test."""
    months = pd.period_range(config.P1_START_MONTH, config.P1_END_MONTH, freq="M").astype(str).tolist()
    weeks = full_weeks(months)
    index = np.arange(len(weeks))
    counts = pd.concat([
        pd.DataFrame({
            "station_id": station,
            "week_start": weeks,
            "departures": (20 * station + 3 * (index % 13)
                           + np.rint(10 * np.sin(2 * np.pi * index / 52))).astype(int),
        })
        for station in (1, 2, 3, 4)
    ], ignore_index=True)
    panel, _ = p1_station_week_panel(
        counts, [1, 2, 3, 4], months, cohort_start_month=config.P1_COHORT_START_MONTH)
    X = candidate_features(panel)
    train = panel.split.eq("train")
    assert X.loc[train, "lag_52"].isna().any()
    assert X.loc[train, "lag_52"].notna().any()
    if missing_prediction_history:
        X.loc[~train, "lag_52"] = np.nan
        assert X.loc[~train, "lag_52"].isna().all()
    assert panel.loc[train, "week_start"].max() < panel.loc[~train, "week_start"].min()

    parameters = json.loads(
        (config.PROJ_ROOT / "configs/p1/selected.json").read_text(encoding="utf-8"))
    first = candidate_specs("hgb", parameters)[0][1]
    second = candidate_specs("hgb", parameters)[0][1]
    assert first is not second
    assert first.random_state == second.random_state == config.RANDOM_SEED

    # Un hilo evita sobrecostes de OpenMP en fixtures pequeños y se restaura al salir.
    with threadpool_limits(limits=1):
        first.fit(X.loc[train], panel.loc[train, "departures"])
        second.fit(X.loc[train], panel.loc[train, "departures"])
        for part in ("validation", "test"):
            mask = panel.split.eq(part)
            actual = panel.loc[mask, "departures"]
            prediction_a = first.predict(X.loc[mask])
            prediction_b = second.predict(X.loc[mask])
            assert np.isfinite(prediction_a).all()
            assert (prediction_a >= 0).all()
            # Evitar que una salida constante trivial pase como modelo ajustado.
            assert np.ptp(prediction_a) > 1
            np.testing.assert_allclose(prediction_a, prediction_b, rtol=1e-12, atol=1e-12)
            assert regression_metrics(actual, prediction_a) == pytest.approx(
                regression_metrics(actual, prediction_b), rel=1e-12, abs=1e-12)
