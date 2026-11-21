"""Modelo sintético para validar infraestructura en CI, no para medir P1."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from mibici import config
from mibici.serving import prepare_model, register_p1, registry_connection


def create(destination):
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    rng = np.random.default_rng(13)
    frame = pd.DataFrame(rng.uniform(1, 100, (80, 5)), columns=config.P1_CANDIDATE_FEATURES)
    phase = np.linspace(0, 2 * np.pi, len(frame))
    frame["week_sin"], frame["week_cos"] = np.sin(phase), np.cos(phase)
    target = frame.lag_1 * .9 + 2
    estimator = HistGradientBoostingRegressor(
        loss="poisson", max_iter=5, max_leaf_nodes=7, min_samples_leaf=5,
        early_stopping=False, random_state=13).fit(frame, target)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.PROJ_ROOT,
                                       text=True).strip()
    with tempfile.TemporaryDirectory() as tmp:
        uri = "sqlite:///" + (Path(tmp) / "fixture.db").as_posix()
        with registry_connection(uri) as client:
            experiment = client.create_experiment(
                "compose-synthetic-fixture", artifact_location=(Path(tmp) / "artifacts").as_uri())
            with mlflow.start_run(experiment_id=experiment, tags={
                "question": "P1", "selected_by_validation": "true", "git_commit": revision,
                "fixture": "synthetic_infrastructure_only",
            }):
                mlflow.log_param("data_version", "synthetic-ci-not-mibici")
                info = mlflow.sklearn.log_model(
                    estimator, name="model", signature=infer_signature(frame, target),
                    input_example=frame.head(2), pip_requirements=str(config.PROJ_ROOT / "requirements.txt"),
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
                    skops_trusted_types=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"])
            identity = register_p1(info.model_uri, config.API_MODEL_NAME, uri)
            return prepare_model(config.API_MODEL_NAME, uri, destination, version=identity["version"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create(args.output), indent=2))
