"""Registry temporal, paquete offline y API; no usa MiBici real ni servidor externo."""

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from mibici import config
from mibici.api import create_app
from mibici.serving import (LoadedP1, load_model_bundle, prepare_model,
                           register_p1, registry_connection)


# MLflow 3.17 usa noload en su ORM con SQLAlchemy 2.1. La excepción es
# específica de esa advertencia de terceros; el resto sigue bajo -W error.
pytestmark = pytest.mark.filterwarnings(
    "ignore:The ``noload`` loader strategy is deprecated:sqlalchemy.exc.SADeprecationWarning")


@pytest.fixture(scope="module")
def registered_bundle(tmp_path_factory):
    import mlflow
    from mlflow.models import infer_signature

    root = tmp_path_factory.mktemp("serving")
    uri = "sqlite:///" + str(root / "tracking.db")
    example = pd.DataFrame({
        "lag_1": [20., 30.], "mean_previous_4": [18., 25.],
        "lag_52": [np.nan, 27.], "week_sin": [0., .5], "week_cos": [1., .866],
    })
    with registry_connection(uri) as client:
        experiment_id = client.create_experiment(
            "api-tests", artifact_location=(root / "artifacts").as_uri())
        with mlflow.start_run(experiment_id=experiment_id, tags={
            "question": "P1", "selected_by_validation": "true", "git_commit": "synthetic-commit",
        }) as run:
            mlflow.log_param("data_version", "synthetic-data")
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=str(config.PROJ_ROOT / "mibici/modeling/persistence_model.py"),
                signature=infer_signature(example, example.lag_1.to_numpy()),
                input_example=example, pip_requirements=["mlflow==" + mlflow.__version__])
        source = info.model_uri
    identity = register_p1(source, "test-p1", uri)
    bundle = root / "prepared"
    prepare_model("test-p1", uri, bundle, version=identity["version"])
    return bundle, uri, identity, source



@pytest.fixture(scope="module")
def api_client(registered_bundle, tmp_path_factory):
    """HTTP real con Uvicorn; usa requests, que ya pertenece al entorno fijo."""
    import os
    import socket
    import subprocess
    import sys
    import time
    import requests

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["MIBICI_MODEL_DIR"] = str(registered_bundle[0])
    env["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    log_path = tmp_path_factory.mktemp("api-log") / "uvicorn.log"
    session = requests.Session()
    session.trust_env = False

    class Client:
        def get(self, path):
            return session.get(base + path, timeout=10)

        def post(self, path, **kwargs):
            return session.post(base + path, timeout=10, **kwargs)

    with log_path.open("w") as stream:
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "mibici.api:app",
             "--host", "127.0.0.1", "--port", str(port)],
            cwd=config.PROJ_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            ready = False
            for _ in range(150):
                if server.poll() is not None:
                    break
                try:
                    ready = session.get(base + "/health", timeout=.5).status_code == 200
                    if ready:
                        break
                except requests.RequestException:
                    pass
                time.sleep(.1)
            assert ready, log_path.read_text()
            yield Client()
        finally:
            session.close()
            if server.poll() is None:
                server.terminate()
            server.wait(timeout=15)


def test_registry_download_pins_alias_and_preserves_lineage(registered_bundle, tmp_path):
    bundle, uri, identity, source = registered_bundle
    assert register_p1(source, "test-p1", uri) == identity
    with registry_connection(uri) as client:
        client.set_registered_model_alias("test-p1", "candidate", identity["version"])
    metadata = prepare_model("test-p1", uri, tmp_path / "alias-copy", alias="candidate")
    assert metadata["registered_uri"] == "models:/test-p1/" + identity["version"]
    assert metadata["run_id"] == identity["run_id"]
    assert metadata["git_commit"] == "synthetic-commit"
    assert metadata["data_version"] == "synthetic-data"
    assert "MLmodel" in metadata["model_files"]


def test_loaded_bundle_predicts_without_registry(registered_bundle, monkeypatch):
    import requests
    from urllib.request import OpenerDirector

    def offline(*args, **kwargs):
        raise AssertionError("La carga del paquete no debe acceder a la red.")
    monkeypatch.setattr(requests.Session, "request", offline)
    monkeypatch.setattr(OpenerDirector, "open", offline)
    predictor = load_model_bundle(registered_bundle[0])
    example = json.loads((config.PROJ_ROOT / "examples/p1-predict.json").read_text())
    assert predictor.predict(example["instances"]) == [100., 35.]


def test_bundle_tampering_is_rejected_before_model_load(registered_bundle, tmp_path):
    directory = tmp_path / "altered"
    shutil.copytree(registered_bundle[0], directory)
    with (directory / "model/MLmodel").open("a") as stream:
        stream.write("\n# modified")
    with pytest.raises(ValueError, match="modificado"):
        load_model_bundle(directory)


def test_prepare_does_not_replace_existing_bundle(registered_bundle):
    bundle, uri, identity, _ = registered_bundle
    before = (bundle / "registry.json").read_bytes()
    with pytest.raises(ValueError, match="ya existe"):
        prepare_model("test-p1", uri, bundle, version=identity["version"])
    assert (bundle / "registry.json").read_bytes() == before


@pytest.mark.parametrize("version,alias", [(None, None), ("1", "candidate"), ("0", None)])
def test_prepare_requires_exactly_one_valid_version_or_alias(tmp_path, version, alias):
    with pytest.raises(ValueError):
        prepare_model("test-p1", "http://invalid", tmp_path / "model",
                      version=version, alias=alias)


def test_api_predict_health_and_openapi(registered_bundle, api_client):
    payload = json.loads((config.PROJ_ROOT / "examples/p1-predict.json").read_text())
    client = api_client
    assert client.get("/health").json()["status"] == "ok"
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predictions"] == [100., 35.]
    assert data["model"] == registered_bundle[2]
    assert client.get("/openapi.json").json()["paths"]["/predict"]["post"]
    assert client.get("/docs").status_code == 200

@pytest.mark.parametrize("payload", [
    {"instances": []},
    {"instances": [{"lag_1": 1}]},
    {"instances": [{"lag_1": "bad", "mean_previous_4": 1, "week_sin": 0, "week_cos": 1}]},
])
def test_api_rejects_malformed_payloads(api_client, payload):
    assert api_client.post("/predict", json=payload).status_code == 422


def test_api_fails_startup_without_a_model(tmp_path):
    import asyncio

    async def startup():
        app = create_app(tmp_path / "absent")
        async with app.router.lifespan_context(app):
            pass

    with pytest.raises(FileNotFoundError):
        asyncio.run(startup())


@pytest.mark.parametrize("values", [[np.nan], [-1.], [1., 2.], [[1.]]])
def test_invalid_model_output_is_not_returned(values):
    class BrokenModel:
        def predict(self, frame):
            return values
    with pytest.raises(RuntimeError):
        LoadedP1(BrokenModel(), {}).predict([{}])
