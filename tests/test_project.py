"""Contrato de ejecución, parametrización y separación temporal de entrega 2."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mibici import config
from mibici.dataset import DataError
from mibici.modeling.p1_candidates import candidate_features, candidate_specs
from mibici.project import read_parameters


def test_mlproject_parameters_resolve_defaults_in_config():
    project = yaml.safe_load((config.PROJ_ROOT / "MLproject").read_text())
    assert {"ingest", "train"} <= set(project["entry_points"])
    for entry in project["entry_points"].values():
        assert all(spec["default"] == "config" for spec in entry["parameters"].values())
    environment = yaml.safe_load((config.PROJ_ROOT / project["python_env"]).read_text())
    assert environment["dependencies"] == ["-r requirements.txt", "-e ."]
    assert environment["python"].startswith("3.12.")


def test_p1_candidate_features_never_include_current_target():
    frame = pd.DataFrame({"week_start": pd.to_datetime(["2026-01-05", "2026-01-12"]),
                          "lag_1": [5., 7.], "mean_previous_4": [4., 6.],
                          "lag_52": [np.nan, 8.], "departures": [7, 10]})
    before = candidate_features(frame)
    frame["departures"] = [10000, 20000]
    pd.testing.assert_frame_equal(before, candidate_features(frame))
    assert before.columns.tolist() == config.P1_CANDIDATE_FEATURES
    assert before.week_sin.between(-1, 1).all()
    assert before.lag_52.isna().iloc[0]


def test_three_candidates_and_reproducible_winner_configuration():
    specs = candidate_specs("hgb-search", {})
    assert len(specs) >= 3
    assert len({model.max_leaf_nodes for _, model in specs}) == 3
    for _, model in specs:
        assert model.early_stopping is False
        assert model.random_state == config.RANDOM_SEED
    single = candidate_specs("hgb", {"max_leaf_nodes": 7, "max_iter": 20})[0][1]
    assert single.max_leaf_nodes == 7 and single.max_iter == 20
    assert candidate_specs("baseline", {}) == []


@pytest.mark.parametrize("model,params", [("rf", {}), ("hgb", {"early_stopping": True}),
                                          ("hgb", {"loss": "squared_error"}),
                                          ("baseline", {"max_iter": 2})])
def test_reject_incompatible_candidate_configuration(model, params):
    with pytest.raises(DataError):
        candidate_specs(model, params)


def test_parameter_files_are_json_objects(tmp_path):
    assert read_parameters("config") == {}
    path = tmp_path / "parameters.json"
    path.write_text(json.dumps({"max_leaf_nodes": 7}))
    assert read_parameters(str(path)) == {"max_leaf_nodes": 7}
    path.write_text("[]")
    with pytest.raises(DataError):
        read_parameters(str(path))


def test_dispatch_passes_parameters_to_p2_p3(monkeypatch, tmp_path):
    from mibici.modeling import train, p2, p3
    captured = []
    monkeypatch.setattr(p2, "run_p2", lambda **kwargs: captured.append(kwargs))
    monkeypatch.setattr(p3, "run_p3", lambda **kwargs: captured.append(kwargs))
    train.dispatch("p2", "random_forest", {"n_estimators": 5}, tmp_path / "m.json",
                   tmp_path, tmp_path, "parent", "commit")
    train.dispatch("p3", "kmeans", {"n_clusters": 3}, tmp_path / "m.json",
                   tmp_path, tmp_path, "parent", "commit")
    assert captured[0]["model_parameters"] == {"n_estimators": 5}
    assert captured[1]["model_parameters"] == {"n_clusters": 3}
    with pytest.raises(DataError):
        train.dispatch("p3", "unknown", {}, tmp_path, tmp_path, tmp_path, "p", "c")
