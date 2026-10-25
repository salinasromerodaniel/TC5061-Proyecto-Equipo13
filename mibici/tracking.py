"""Contrato de tracking de los puntos de entrada de MLflow Projects."""

from contextlib import contextmanager
from importlib import metadata
import os
from pathlib import Path
import sys

from packaging.requirements import Requirement

from mibici import config
from mibici.dataset import DataError


def validate_environment():
    """El modo local debe respetar el entorno declarado, también en Windows."""
    if sys.version_info[:2] != (3, 12):
        raise DataError("El proyecto requiere Python 3.12; consulta python_env.yaml.")
    errors = []
    for line in (config.PROJ_ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            errors.append(f"{requirement.name}: ausente")
            continue
        if version not in requirement.specifier:
            errors.append(f"{requirement.name}: {version} fuera de {requirement.specifier}")
    if errors:
        raise DataError("Instala requirements.txt antes de usar --env-manager=local: "
                        + "; ".join(errors))


def configure_tracking():
    import mlflow
    uri = os.getenv("MLFLOW_TRACKING_URI", config.PROJECT_TRACKING_URI)
    if not uri.startswith(("http://", "https://")):
        raise DataError("MLproject requiere MLFLOW_TRACKING_URI hacia el servidor HTTP local.")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.set_experiment(config.PROJECT_EXPERIMENT)
    return uri


@contextmanager
def project_run(question, model, commit, stage="search"):
    """Retomar el run creado por mlflow run antes de abrir hijos explícitos."""
    import mlflow
    configure_tracking()
    with mlflow.start_run(run_name=f"{question.upper()}_{model}") as parent:
        mlflow.set_tags({"team": "13", "question": question.upper(), "model_family": model,
                         "stage_of_work": stage, "git_commit": commit,
                         "mlflow.source.git.commit": commit})
        mlflow.log_params({"resolved_question": question, "resolved_model": model,
                           "python_version": sys.version.split()[0]})
        for filename in ("MLproject", "python_env.yaml", "requirements.txt", "mibici/config.py"):
            mlflow.log_artifact(str(config.PROJ_ROOT / filename), artifact_path="config")
        yield parent
