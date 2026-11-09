"""Recuperación de P1 por nombre/versión del Registry y carga de un paquete local."""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

from mibici import config


@contextmanager
def registry_connection(uri):
    import mlflow
    from mlflow.tracking import MlflowClient

    previous = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    try:
        yield MlflowClient(tracking_uri=uri, registry_uri=uri)
    finally:
        mlflow.set_tracking_uri(previous[0])
        mlflow.set_registry_uri(previous[1])


def validate_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
        raise ValueError("Nombre de modelo inválido.")
    return name


def validate_signature(model):
    schema = model.metadata.get_input_schema()
    if schema is None or schema.input_names() != config.P1_CANDIDATE_FEATURES:
        raise ValueError("La firma del modelo no corresponde a las cinco variables de P1.")


def register_p1(source, name, tracking_uri):
    """Registrar una corrida ya entrenada; conservar su selección y su linaje."""
    import mlflow

    validate_name(name)
    with registry_connection(tracking_uri) as client:
        info = mlflow.models.get_model_info(source)
        if not info.run_id:
            raise ValueError("El modelo debe conservar su corrida de origen.")
        run = client.get_run(info.run_id)
        if (run.info.status != "FINISHED" or run.data.tags.get("question") != "P1"
                or run.data.tags.get("selected_by_validation") != "true"):
            raise ValueError("Se requiere una corrida P1 finalizada y elegida por validación.")
        commit = run.data.tags.get("git_commit")
        data_version = run.data.params.get("data_version")
        if not commit or not data_version:
            raise ValueError("La corrida no conserva commit y versión de datos.")
        validate_signature(mlflow.pyfunc.load_model(source))
        tags = {"question": "P1", "git_commit": commit, "data_version": data_version,
                "source_run_id": info.run_id}
        versions = client.search_model_versions(f"name = '{name}'")
        version = next((v for v in versions if v.run_id == info.run_id), None)
        if version is None:
            version = mlflow.register_model(source, name, tags=tags)
        for key, value in tags.items():
            client.set_model_version_tag(name, version.version, key, value)
        description = ("Demanda semanal por estación de Zapopan. Elegido por MAE de "
                       f"validación. Run {info.run_id}; commit {commit}; datos {data_version}. "
                       "No estima inventario ni demanda no atendida.")
        if not client.get_registered_model(name).description:
            client.update_registered_model(name, description=description)
        client.update_model_version(name, version.version, description=description)
        return {"name": name, "version": str(version.version), "run_id": info.run_id,
                "git_commit": commit, "data_version": data_version}


def model_hashes(directory):
    result = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_symlink():
            raise ValueError("El paquete no admite enlaces simbólicos.")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    return result


def prepare_model(name, tracking_uri, destination, *, version=None, alias=None):
    """Resolver alias una vez y descargar una versión fija, sin sobrescribir paquetes."""
    import mlflow

    validate_name(name)
    if bool(version) == bool(alias):
        raise ValueError("Indica exactamente una versión o un alias.")
    if version and (not str(version).isdigit() or int(version) < 1):
        raise ValueError("La versión debe ser un entero positivo.")
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("El destino ya existe; usa otro directorio para otra versión.")
    with registry_connection(tracking_uri) as client:
        registered = (client.get_model_version_by_alias(name, alias) if alias
                      else client.get_model_version(name, str(version)))
        if registered.status != "READY" or not registered.run_id:
            raise ValueError("La versión registrada no está lista o carece de corrida.")
        run = client.get_run(registered.run_id)
        if run.data.tags.get("question") != "P1":
            raise ValueError("La versión registrada no pertenece a P1.")
        identity = {"name": name, "version": str(registered.version),
                    "registered_uri": f"models:/{name}/{registered.version}",
                    "run_id": registered.run_id,
                    "git_commit": run.data.tags.get("git_commit"),
                    "data_version": run.data.params.get("data_version")}
        if not identity["git_commit"] or not identity["data_version"]:
            raise ValueError("Falta el linaje de commit o de datos.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=destination.parent) as temp:
            temp = Path(temp)
            downloaded = mlflow.artifacts.download_artifacts(
                artifact_uri=identity["registered_uri"], dst_path=str(temp / "download"))
            bundle = temp / "bundle"
            shutil.copytree(downloaded, bundle / "model")
            model = mlflow.pyfunc.load_model(str(bundle / "model"))
            validate_signature(model)
            identity["model_files"] = model_hashes(bundle / "model")
            (bundle / "registry.json").write_text(
                json.dumps(identity, indent=2) + "\n", encoding="utf-8")
            bundle.rename(destination)
    return identity


@dataclass
class LoadedP1:
    model: object
    identity: dict

    def predict(self, instances):
        predictions = np.asarray(self.model.predict(pd.DataFrame(instances, dtype=float)), dtype=float)
        if (predictions.ndim != 1 or len(predictions) != len(instances)
                or not np.isfinite(predictions).all() or (predictions < 0).any()):
            raise RuntimeError("El modelo devolvió una salida de predicción inválida.")
        return predictions.tolist()


def load_model_bundle(directory):
    """Comprobar integridad antes de cargar; no necesita acceso al Registry."""
    import mlflow

    directory = Path(directory)
    identity = json.loads((directory / "registry.json").read_text(encoding="utf-8"))
    for key in ("name", "version", "run_id", "git_commit", "data_version"):
        if not isinstance(identity.get(key), str) or not identity[key]:
            raise ValueError(f"El paquete no contiene {key}.")
    expected_uri = f"models:/{identity['name']}/{identity['version']}"
    if identity.get("registered_uri") != expected_uri:
        raise ValueError("El paquete no identifica una versión fija del Registry.")
    hashes = model_hashes(directory / "model")
    if not hashes or hashes != identity.get("model_files") or "MLmodel" not in hashes:
        raise ValueError("El modelo local está incompleto o fue modificado.")
    model = mlflow.pyfunc.load_model(str(directory / "model"))
    validate_signature(model)
    return LoadedP1(model, {key: identity[key] for key in (
        "name", "version", "run_id", "git_commit", "data_version")})


def main(argv=None):
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    source_choice = register.add_mutually_exclusive_group(required=True)
    source_choice.add_argument("--source", help="URI del modelo ya entrenado")
    source_choice.add_argument("--handoff", type=Path, help="Resultado de una búsqueda P1")
    download = commands.add_parser("download")
    choice = download.add_mutually_exclusive_group(required=True)
    choice.add_argument("--version")
    choice.add_argument("--alias")
    download.add_argument("--output", type=Path, default=config.API_MODEL_DIR)
    for entry in (register, download):
        entry.add_argument("--name", default=config.API_MODEL_NAME)
        entry.add_argument("--tracking-uri",
                           default=os.getenv("MLFLOW_TRACKING_URI", config.PROJECT_TRACKING_URI))
    args = parser.parse_args(argv)
    if args.command == "register":
        source = args.source
        if args.handoff:
            handoff = json.loads(args.handoff.read_text(encoding="utf-8"))
            source = handoff.get("model_uri")
            if not source:
                raise ValueError("El handoff no contiene model_uri.")
        result = register_p1(source, args.name, args.tracking_uri)
    else:
        result = prepare_model(args.name, args.tracking_uri, args.output,
                               version=args.version, alias=args.alias)
        result = {key: value for key, value in result.items() if key != "model_files"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(f"Error: {error}") from error
