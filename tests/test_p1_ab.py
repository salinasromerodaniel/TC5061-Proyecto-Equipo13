from dataclasses import replace

import pandas as pd
import pytest

from mibici.dataset import DataError
from mibici.monitoring import FEATURES
from mibici.p1_ab import EXPERIMENT, evaluate, load_registered, save_evaluation, train_challenger
from mibici.serving import registry_connection

pytestmark = pytest.mark.filterwarnings(
    "ignore:The ``noload`` loader strategy is deprecated:sqlalchemy.exc.SADeprecationWarning"
)


def test_registered_challenger_and_paired_evaluation(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    dates = pd.date_range("2026-01-05", periods=9, freq="W-MON")
    frame = pd.DataFrame([
        [station, week, 10 * station + index, float(10 * station + index - 1),
         float(10 * station), float(station) if station > 3 else float("nan"), 0.0, 1.0]
        for index, week in enumerate(dates)
        for station in range(1, 21)
    ], columns=["station_id", "week_start", "departures", *FEATURES])
    reference, validation, current = frame.iloc[:100], frame.iloc[100:120], frame.iloc[120:]
    contract = {"features": FEATURES, "target": "departures", "contract_id": "fixture",
                "historical_data_version": "fixture-train", "keys": ["station_id", "week_start"],
                "partitions": {"current": {"rows": len(current)}},
                "model_a": {"name": "fixture-p1"}}
    with registry_connection(uri) as client:
        client.create_experiment(EXPERIMENT, artifact_location=(tmp_path / "artifacts").as_uri())
        first = train_challenger(reference, validation, contract)
        initial = load_registered(client, first.name, str(first.version), contract, is_a=False)
        contract["model_a"] = initial.identity
        a = load_registered(client, first.name, str(first.version), contract, is_a=True)
        wrong_contract = {**contract, "model_a": {**initial.identity, "model_sha256": "incorrecto"}}
        with pytest.raises(DataError, match="SHA-256"):
            load_registered(client, first.name, str(first.version), wrong_contract, is_a=True)

        second = train_challenger(reference, validation, contract)
        b = load_registered(client, first.name, str(second.version), contract, is_a=False)
        assert first.name == second.name and first.version != second.version
        result = evaluate(current, contract, a, b)
        assert result["metrics"]["A"] == result["metrics"]["B"]
        assert result["recommended_version"] == str(first.version)

        class ExactModel:
            def predict(self, inputs):
                assert inputs.equals(current[FEATURES])
                return current.departures.to_numpy()

        result = evaluate(current, contract, a, replace(b, model=ExactModel()))
        assert result["metrics"]["B"]["mae"] == 0
        assert result["recommended_version"] == str(second.version)
        assert len(result["weekly"]) == 3
        save_evaluation(result, tmp_path / "evaluation")
        comparison = pd.read_csv(tmp_path / "evaluation/comparison.csv")
        assert comparison.arm.tolist() == ["A", "B"]
        assert comparison.mae.iloc[1] == 0
