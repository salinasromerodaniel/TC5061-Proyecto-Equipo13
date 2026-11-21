"""Importación real en Registry temporal, persistencia y protección de identidad."""
import pytest

from mibici.operations import import_bundle
from mibici.serving import load_model_bundle, model_hashes, registry_connection

pytestmark = pytest.mark.filterwarnings(
    "ignore:The ``noload`` loader strategy is deprecated:sqlalchemy.exc.SADeprecationWarning")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    from sklearn.ensemble import HistGradientBoostingRegressor
    import mlflow
    import mlflow.sklearn
    import pandas as pd
    import numpy as np
    from mlflow.models import infer_signature
    from mibici import config
    from mibici.serving import prepare_model, register_p1

    root = tmp_path_factory.mktemp("operation-source")
    frame = pd.DataFrame(np.arange(100, dtype=float).reshape(20, 5),
                         columns=config.P1_CANDIDATE_FEATURES)
    estimator = HistGradientBoostingRegressor(
        max_iter=2, min_samples_leaf=2, random_state=13).fit(frame, frame.lag_1 + 1)
    uri = "sqlite:///" + (root / "source.db").as_posix()
    with registry_connection(uri) as client:
        experiment = client.create_experiment(
            "source", artifact_location=(root / "artifacts").as_uri())
        with mlflow.start_run(experiment_id=experiment, tags={
            "question": "P1", "selected_by_validation": "true", "git_commit": "fixture-source",
        }):
            mlflow.log_param("data_version", "fixture-source-data")
            info = mlflow.sklearn.log_model(
                estimator, name="model", signature=infer_signature(frame, frame.lag_1 + 1),
                input_example=frame.head(2), pip_requirements=[],
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
                skops_trusted_types=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"])
        identity = register_p1(info.model_uri, "fixture-p1", uri)
        path = root / "bundle"
        prepare_model("fixture-p1", uri, path, version=identity["version"])
    return path, model_hashes(path / "model")["model.skops"], frame


def test_import_preserves_predictions_and_does_not_duplicate_versions(source, tmp_path):
    bundle, sha, frame = source
    uri = "sqlite:///" + (tmp_path / "target.db").as_posix()
    with registry_connection(uri) as client:
        client.create_experiment("mibici-equipo13-operation",
                                 artifact_location=(tmp_path / "artifacts").as_uri())
    destination = tmp_path / "prepared"
    first = import_bundle(bundle, uri, sha, destination, "fixture-operation")
    second = import_bundle(bundle, uri, sha, destination, "fixture-operation")
    assert first == second
    assert first["source"]["run_id"] != first["registered"]["run_id"]
    assert first["retrained"] is False
    assert first["registered"]["git_commit"] == "fixture-source"
    assert load_model_bundle(bundle).predict(frame) == load_model_bundle(destination).predict(frame)
    with registry_connection(uri) as client:
        assert len(client.search_model_versions("name = 'fixture-p1'")) == 1
        run = client.get_run(first["registered"]["run_id"])
        assert run.data.tags["source_run_id"] == first["source"]["run_id"]
        assert run.data.tags["operation_code_revision"] == "fixture-operation"


def test_wrong_source_hash_fails_without_importing(source, tmp_path):
    bundle, _, _ = source
    with pytest.raises(ValueError, match="SHA256"):
        import_bundle(bundle, "sqlite:///" + (tmp_path / "unused.db").as_posix(),
                      "wrong-hash", tmp_path / "model")
    assert not (tmp_path / "unused.db").exists()
