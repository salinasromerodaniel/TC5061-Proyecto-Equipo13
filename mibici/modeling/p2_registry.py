"""Registrar exclusivamente el ganador de P2 y comprobar su alias de producción."""

import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

NAME = "mibici-equipo13-p2-riesgo"
DESCRIPTION = (
    "Regresión logística con estandarización para priorizar la revisión de estaciones "
    "de Zapopan por déficit neto semanal indirecto de bicicletas. Seleccionada por "
    "F1 de validación, considerando estabilidad y costo. No estima inventario ni "
    "episodios reales de estaciones vacías o anclajes llenos."
)


def main():
    # Leer las rutas y el servidor indicados al ejecutar el script.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("reports/delivery2/p2/results.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/delivery2/p2/registry.json")
    )
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    args = parser.parse_args()

    # Usar el mismo servidor para consultar experimentos y registrar modelos.
    if not args.tracking_uri or not args.tracking_uri.startswith(
        ("http://", "https://")
    ):
        parser.error("Configura MLFLOW_TRACKING_URI hacia el servidor HTTP del equipo.")
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_registry_uri(args.tracking_uri)
    client = MlflowClient()

    # Recuperar el ganador elegido en la Parte B; aquí no se reentrena.
    results = json.loads(args.results.read_text())
    selected = next(
        row
        for row in results["models"]
        if row["model"] == results["selected_by_validation_f1"]
    )
    if selected["model_family"] != "logistic":
        raise ValueError(
            "Este registro corresponde a la logística seleccionada para P2."
        )

    # Comprobar que los datos, el código y la corrida coinciden con los resultados.
    run = client.get_run(selected["run_id"])
    experiment = client.get_experiment(run.info.experiment_id)
    if (
        run.info.status != "FINISHED"
        or run.data.tags.get("question") != "P2"
        or experiment.name != "mibici-equipo13"
        or run.data.tags.get("git_commit") != results["git_commit"]
        or run.data.params.get("data_version") != results["data_version"]
    ):
        raise ValueError("La corrida no coincide con el linaje de los resultados P2.")
    model_uri = selected["model_uri"]
    logged = client.get_logged_model(model_uri.removeprefix("models:/"))
    if logged.source_run_id != run.info.run_id:
        raise ValueError("El modelo no pertenece a la corrida seleccionada.")

    # El ejemplo guardado permite comprobar la recuperación sin cargar los viajes.
    example_path = mlflow.artifacts.download_artifacts(
        artifact_uri=logged.artifact_location + "/input_example.json"
    )
    example = pd.read_json(example_path, orient="split")
    original = mlflow.sklearn.load_model(model_uri)
    expected = original.predict(example)
    expected_proba = original.predict_proba(example)

    # Conservar en el Registry el origen y el criterio de selección del modelo.
    tags = {
        "question": "P2",
        "team": "13",
        "model_family": "logistic",
        "git_commit": results["git_commit"],
        "data_version": results["data_version"],
        "experiment": experiment.name,
        "source_run_id": run.info.run_id,
        "selection_metric": "validation_f1",
        "target_kind": "indirect_proxy",
    }

    # Reutilizar versiones existentes y evitar sustituir otro champion por accidente.
    versions = []
    try:
        registered = client.get_registered_model(NAME)
        versions = list(client.search_model_versions(f"name='{NAME}'"))
        champion = registered.aliases.get("champion")
        if champion:
            current = client.get_model_version(NAME, champion)
            if current.run_id != run.info.run_id:
                raise ValueError(
                    "champion ya apunta a otra corrida; revisar antes de sustituirlo."
                )
    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise

    # Crear una versión solo si esta corrida todavía no está registrada.
    matches = [
        v
        for v in versions
        if v.run_id == run.info.run_id
        and v.source in (logged.artifact_location, model_uri)
    ]
    version = (
        max(matches, key=lambda v: int(v.version))
        if matches
        else mlflow.register_model(model_uri, NAME, tags=tags)
    )

    # Documentar la tarea y sus límites en el modelo y en su versión.
    client.update_registered_model(NAME, description=DESCRIPTION)
    client.update_model_version(
        NAME,
        version.version,
        description=DESCRIPTION
        + " Producción académica mediante alias champion; no despliegue operativo.",
    )
    for key, value in tags.items():
        client.set_model_version_tag(NAME, version.version, key, value)
    # Comprobar la versión antes de asignar producción.
    restored = mlflow.sklearn.load_model(f"models:/{NAME}/{version.version}")
    np.testing.assert_array_equal(expected, restored.predict(example))
    np.testing.assert_allclose(expected_proba, restored.predict_proba(example))

    # Asignar el alias después de verificar la carga por número de versión.
    client.set_registered_model_alias(NAME, "champion", version.version)
    alias_uri = f"models:/{NAME}@champion"

    # Confirmar que el nombre de producción recupera las mismas predicciones.
    production = mlflow.sklearn.load_model(alias_uri)
    np.testing.assert_array_equal(expected, production.predict(example))
    np.testing.assert_allclose(expected_proba, production.predict_proba(example))
    client.set_model_version_tag(NAME, version.version, "alias_load_verified", "true")

    # Guardar evidencia técnica; el reporte presenta una explicación en lenguaje claro.
    evidence = {
        "name": NAME,
        "version": str(version.version),
        "alias": "champion",
        "model_uri": alias_uri,
        "tracking_uri": args.tracking_uri,
        "experiment": experiment.name,
        "run_id": run.info.run_id,
        "git_commit": results["git_commit"],
        "data_version": results["data_version"],
        "source_model_uri": model_uri,
        "alias_load_verified": True,
        "verification_rows": len(example),
        "predictions_equal": True,
        "probabilities_equal": True,
        "scope": "P2 only; local academic registry, not operational deployment",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
