"""Aprovisionamiento del modelo P1 para la operación con Compose."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from mibici.serving import load_model_bundle, prepare_model, registry_connection, validate_name

EXPERIMENT = "mibici-equipo13-operation"


def init_volumes():
    """Sólo el contenedor efímero ejecuta esta preparación como root."""
    for name in ("/state", "/models"):
        path = Path(name)
        path.mkdir(parents=True, exist_ok=True)
        os.chown(path, 10001, 10001)
        path.chmod(0o750)


def source_identity(bundle, expected_sha):
    loaded = load_model_bundle(bundle)
    full = json.loads((Path(bundle) / "registry.json").read_text(encoding="utf-8"))
    actual = full["model_files"].get("model.skops")
    if not expected_sha or actual != expected_sha:
        raise ValueError("El modelo de origen no coincide con MIBICI_EXPECTED_MODEL_SHA256.")
    identity = {**loaded.identity, "model_sha256": actual}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity, fingerprint


def import_bundle(bundle, tracking_uri, expected_sha, destination, revision="working-tree"):
    """Importar el artefacto entrenado, sin reentrenarlo ni atribuirle un run original nuevo."""
    origin, fingerprint = source_identity(bundle, expected_sha)
    name = validate_name(origin["name"])
    with registry_connection(tracking_uri) as client:
        versions = client.search_model_versions(f"name = '{name}'")
        version = next((v for v in versions if v.tags.get("source_bundle_id") == fingerprint), None)
        if version is None:
            experiment = client.get_experiment_by_name(EXPERIMENT)
            experiment_id = (experiment.experiment_id if experiment
                             else client.create_experiment(EXPERIMENT))
            run = client.create_run(experiment_id, tags={
                "question": "P1", "stage_of_work": "model_import_no_training",
                "git_commit": origin["git_commit"], "operation_code_revision": revision,
                "source_run_id": origin["run_id"], "source_model_version": origin["version"],
                "source_bundle_id": fingerprint,
            })
            run_id = run.info.run_id
            try:
                client.log_param(run_id, "data_version", origin["data_version"])
                client.log_param(run_id, "model_sha256", expected_sha)
                client.log_artifacts(run_id, str(Path(bundle) / "model"), "model")
                client.set_terminated(run_id, "FINISHED")
                if not versions:
                    # El nombre puede existir sin versiones; no crearlo otra vez.
                    from mlflow.exceptions import MlflowException
                    try:
                        client.create_registered_model(name)
                    except MlflowException as error:
                        if error.error_code != "RESOURCE_ALREADY_EXISTS":
                            raise
                version = client.create_model_version(
                    name, f"{run.info.artifact_uri}/model", run_id=run_id,
                    tags={"source_bundle_id": fingerprint, "source_run_id": origin["run_id"]},
                    description="Importación para Compose, sin entrenamiento. "
                                f"Modelo original: run {origin['run_id']}, versión {origin['version']}.")
            except Exception:
                client.set_terminated(run_id, "FAILED")
                raise
        if version.status != "READY":
            raise ValueError("La versión importada aún no está disponible.")
        destination = Path(destination)
        if destination.exists():
            cached = load_model_bundle(destination)
            if (cached.identity["run_id"] != version.run_id
                    or cached.identity["version"] != str(version.version)):
                raise ValueError("El volumen contiene otra versión; usa un proyecto Compose separado.")
            source_identity(destination, expected_sha)
        else:
            prepare_model(name, tracking_uri, destination, version=str(version.version))
            source_identity(destination, expected_sha)
        provenance = {"source": origin, "source_bundle_id": fingerprint,
                      "registered": load_model_bundle(destination).identity,
                      "operation_code_revision": revision, "retrained": False}
        (destination / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def provision():
    uri = os.environ["MLFLOW_TRACKING_URI"]
    remote = os.environ.get("MIBICI_SOURCE_REGISTRY", "")
    expected = os.environ.get("MIBICI_EXPECTED_MODEL_SHA256", "")
    with tempfile.TemporaryDirectory() as tmp:
        source = Path("/import/p1")
        if remote:
            source = Path(tmp) / "source"
            prepare_model(os.environ.get("MIBICI_SOURCE_MODEL_NAME", "mibici-equipo13-p1-demanda"),
                          remote, source, version=os.environ.get("MIBICI_SOURCE_MODEL_VERSION", "1"))
        elif not (source / "registry.json").is_file():
            raise ValueError("Falta el modelo: prepara MIBICI_SOURCE_BUNDLE o configura "
                             "MIBICI_SOURCE_REGISTRY y una versión accesible desde el contenedor.")
        return import_bundle(source, uri, expected, "/models/p1",
                             os.environ.get("MIBICI_CODE_REVISION", "working-tree"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("volumes", "provision"))
    args = parser.parse_args(argv)
    try:
        if args.command == "volumes":
            init_volumes()
        else:
            print(json.dumps(provision(), indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
