"""P2: déficit neto semanal indirecto; nunca stockouts observados.

Parte B: mlflow run . --env-manager=local --experiment-name mibici-equipo13 -e train -P question=p2 -P model=search
"""

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mibici import config
from mibici.dataset import read_manifest, verify_manifest
from mibici.features import full_weeks
from mibici.geography import load_boundary, municipality_station_ids
from mibici.modeling.p1 import code_commit
from mibici.p2_panel import build_panel

TARGET = "next_week_departures_minus_arrivals_ge_positive_reference_capacity"


def weekly_counts(manifest, data_dir):
    """Sumar llegadas y salidas por sus fechas respectivas en lotes mensuales."""
    parts = []
    for entry in manifest["files"]:
        if entry["kind"] != "trips":
            continue
        path = Path(data_dir) / entry["path"]
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=config.CHUNK_SIZE,
            columns=[
                "started_at",
                "ended_at",
                "origin_station_id",
                "destination_station_id",
            ],
        ):
            trips = batch.to_pandas()
            for station, time, name in (
                ("origin_station_id", "started_at", "departures"),
                ("destination_station_id", "ended_at", "arrivals"),
            ):
                frame = trips[[station, time]].dropna().copy()
                frame["week_start"] = frame[time].dt.normalize() - pd.to_timedelta(
                    frame[time].dt.dayofweek, unit="D"
                )
                frame = frame.rename(columns={station: "station_id"})
                parts.append(
                    frame.groupby(["station_id", "week_start"]).size().rename(name)
                )
    if not parts:
        raise ValueError("El manifiesto no contiene viajes.")
    # Concatenar por filas conserva el nombre de cada medida como columna.
    return (
        pd.concat([part.to_frame() for part in parts])
        .groupby(level=[0, 1])
        .sum()
        .reset_index()
    )


def classification_metrics(actual, predictions, scores):
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
    )

    result = {
        "accuracy": float(accuracy_score(actual, predictions)),
        "precision": float(precision_score(actual, predictions, zero_division=0)),
        "recall": float(recall_score(actual, predictions, zero_division=0)),
        "f1": float(f1_score(actual, predictions, zero_division=0)),
        "positive_prevalence": float(actual.mean()),
        "positive_support": int(actual.sum()),
        "rows": len(actual),
    }
    if actual.sum() > 0:
        result["average_precision"] = float(average_precision_score(actual, scores))
    return result


def performance_figure(actual, prediction, scores, path, title):
    """Matriz de confusión y curva precisión-recall del mismo conjunto."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(
        actual,
        prediction,
        labels=[0, 1],
        display_labels=["Sin proxy", "Con proxy"],
        ax=axes[0],
        colorbar=False,
    )
    PrecisionRecallDisplay.from_predictions(actual, scores, ax=axes[1])
    axes[1].axhline(
        float(actual.mean()), color="grey", linestyle="--", label="Prevalencia"
    )
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_model(family, parameters):
    """Construir únicamente los tres tipos de modelo usados por P2."""
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if family == "dummy_most_frequent":
        model = DummyClassifier(strategy="most_frequent")
    elif family == "random_forest":
        model = RandomForestClassifier(**config.P2_MODEL_DEFAULTS)
    elif family == "logistic":
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(**config.P2_LOGISTIC_DEFAULTS)),
            ]
        )
    else:
        raise ValueError(f"Modelo P2 desconocido: {family}.")
    if any(key.endswith("random_state") for key in parameters):
        raise ValueError("La semilla P2 se configura en config.py.")
    return model.set_params(**parameters)


def run_p2(
    manifest_path=config.P2_DATA_MANIFEST,
    data_dir=config.DATA_DIR,
    tracking_uri=None,
    report_dir=config.P2_REPORT_DIR,
    boundary_manifest=config.P1_BOUNDARY_MANIFEST,
    model_name=None,
    model_parameters=None,
):
    """P2: baseline y candidatas comparables; search se anida en el run de Project."""
    import mlflow
    import mlflow.sklearn
    import sklearn
    from mlflow.models import infer_signature
    from sklearn.metrics import classification_report, confusion_matrix

    # Verificar las fuentes antes de construir variables o entrenar.
    commit = code_commit()
    manifest = read_manifest(manifest_path)
    verify_manifest(manifest, data_dir)

    # Preparar el panel municipal conservando las particiones originales.
    counts = weekly_counts(manifest, data_dir)
    stations = pd.read_parquet(Path(data_dir) / "processed" / "stations.parquet")
    geometry, geography = load_boundary(data_dir, boundary_manifest)
    zapopan_ids = municipality_station_ids(stations, geometry)
    if not zapopan_ids:
        raise ValueError("El polígono de Zapopan no contiene estaciones del catálogo.")
    # Filtrar estaciones después de contar ambos extremos de todos los viajes:
    # un viaje que cruza el municipio sí aporta al extremo situado en Zapopan.
    panel, features, metadata = build_panel(
        counts,
        stations,
        full_weeks(manifest["months"]),
        config.P2_VALIDATION_WEEKS,
        config.P2_TEST_WEEKS,
        station_ids=zapopan_ids,
    )
    metadata.update(
        {
            "scope": "inegi_zapopan_reference_polygon_historical_cohort",
            "municipality_code": geography["municipality_code"],
            "municipality_name": geography["municipality_name"],
            "boundary_sha256": geography["sha256"],
            "geographic_station_ids": zapopan_ids,
            "cohort_station_ids": sorted(int(i) for i in panel.station_id.unique()),
        }
    )
    metadata["time_policy"] = manifest["time_policy"]
    # Auditar eventos excluidos por catálogo/capacidad; no inventar capacidad.
    valid_ids = set(stations.loc[stations.capacity > 0, "station_id"])
    excluded = counts[~counts.station_id.isin(valid_ids)]
    metadata["events_excluded_by_catalog_or_capacity"] = {
        name: int(excluded[name].sum()) for name in ("departures", "arrivals")
    }
    outside = counts[
        counts.station_id.isin(stations.station_id)
        & ~counts.station_id.isin(zapopan_ids)
    ]
    metadata["events_outside_zapopan_reference"] = {
        name: int(outside[name].sum()) for name in ("departures", "arrivals")
    }
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    train = panel[panel.split == "train"]
    if train.target.nunique() != 2:
        raise ValueError(
            "Entrenamiento necesita ambas clases; revisar etiqueta y periodo."
        )
    uri = (
        tracking_uri
        or os.getenv("MLFLOW_TRACKING_URI")
        or ("sqlite:///" + str(config.P2_TRACKING_DB.resolve()))
    )
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(config.P2_EXPERIMENT) is None:
        mlflow.create_experiment(
            config.P2_EXPERIMENT,
            artifact_location=config.P2_ARTIFACT_DIR.resolve().as_uri(),
        )
    mlflow.set_experiment(config.P2_EXPERIMENT)
    # La búsqueda usa combinaciones fijadas en config.py para facilitar su comparación.
    if model_name == "search":
        if model_parameters:
            raise ValueError(
                "La búsqueda P2 usa P2_SEARCH_CONFIGS; params_file es para un modelo."
            )
        specs = [("dummy_most_frequent", "dummy_most_frequent", {})]
        specs += [
            (item["name"], item["family"], item["parameters"])
            for item in config.P2_SEARCH_CONFIGS
        ]
    elif model_name is None:
        specs = [
            ("dummy_most_frequent", "dummy_most_frequent", {}),
            ("random_forest", "random_forest", {}),
        ]
    else:
        family = "dummy_most_frequent" if model_name == "baseline" else model_name
        specs = [(family, family, model_parameters or {})]

    # Entrenar cada alternativa con los mismos ejemplos y registrar una corrida hija.
    comparisons = []
    for name, family, parameters in specs:
        model = make_model(family, parameters)
        model_dir = report_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "plots").mkdir(exist_ok=True)
        params = {
            **{
                key: value
                for key, value in model.get_params(deep=True).items()
                if isinstance(value, (str, int, float, bool, type(None)))
            },
            "model": name,
            "seed": config.RANDOM_SEED,
            "data_start_month": manifest["months"][0],
            "data_end_month": manifest["months"][-1],
            "data_version": manifest["data_version"],
            "horizon_days": 7,
            "history_weeks": 4,
            "decision_rule": "estimator_predict_default_no_tuning",
            "target_definition": TARGET,
            "features": ",".join(features),
            "capacity_policy": metadata["capacity_policy"],
            "scope": metadata["scope"],
            "cohort_stations": metadata["cohort_stations"],
            "municipality_code": geography["municipality_code"],
            "municipality_name": geography["municipality_name"],
            "boundary_sha256": geography["sha256"],
            "geographic_catalog_stations": len(zapopan_ids),
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "sklearn_version": sklearn.__version__,
            "mlflow_version": mlflow.__version__,
        }
        for split, details in metadata["partitions"].items():
            params.update({f"{split}_{key}": value for key, value in details.items()})
        with mlflow.start_run(
            run_name=f"P2_Zapopan_{name}_{manifest['months'][0]}_{manifest['months'][-1]}",
            nested=mlflow.active_run() is not None,
            tags={
                "team": "13",
                "question": "P2",
                "git_commit": commit,
                "model_family": family,
                "stage_of_work": (
                    "baseline" if name.startswith("dummy") else "candidate"
                ),
                "mlflow.source.git.commit": commit,
                "target_kind": "indirect_proxy",
                "capacity_reference": "2026-09_retrospective",
                "municipality": "Zapopan",
            },
        ) as run:
            # Ajustar únicamente con entrenamiento; el escalado queda dentro del modelo.
            X_train = train[features].astype(float)
            started = time.perf_counter()
            model.fit(X_train, train.target)
            fit_seconds = time.perf_counter() - started
            mlflow.log_params(params)
            metrics = {"fit_seconds": fit_seconds}
            for split in ("validation", "test"):
                frame = panel[panel.split == split]
                X = frame[features].astype(float)
                started = time.perf_counter()
                prediction = model.predict(X)
                scores = model.predict_proba(X)[:, list(model.classes_).index(1)]
                metrics[f"{split}_predict_seconds"] = time.perf_counter() - started
                if split == "validation":
                    weekly_f1 = [
                        classification_metrics(
                            frame.target.iloc[pos], prediction[pos], scores[pos]
                        )["f1"]
                        for pos in frame.groupby("target_week").indices.values()
                    ]
                    metrics["validation_week_f1_std"] = float(np.std(weekly_f1))
                metrics.update(
                    {
                        f"{split}_{key}": value
                        for key, value in classification_metrics(
                            frame.target, prediction, scores
                        ).items()
                    }
                )
                matrix = confusion_matrix(frame.target, prediction, labels=[0, 1])
                pd.DataFrame(
                    matrix, index=["real_0", "real_1"], columns=["pred_0", "pred_1"]
                ).to_csv(model_dir / f"{split}_confusion_matrix.csv")
                mlflow.log_dict(
                    classification_report(
                        frame.target,
                        prediction,
                        labels=[0, 1],
                        output_dict=True,
                        zero_division=0,
                    ),
                    f"evaluation/{split}_classification_report.json",
                )
                output = frame[
                    ["station_id", "week_start", "target_week", "target"]
                ].copy()
                output["prediction"] = prediction
                output["score"] = scores
                output.to_csv(model_dir / f"{split}_predictions.csv", index=False)
                performance_figure(
                    frame.target,
                    prediction,
                    scores,
                    model_dir / "plots" / f"{split}.png",
                    f"P2 {name} · {split}",
                )

            # Conservar resultados, datos de origen y configuración como artefactos.
            mlflow.log_metrics(metrics)
            mlflow.log_input(
                mlflow.data.from_pandas(
                    X_train, source=str(manifest_path), name="p2_training_features"
                ),
                context="training",
            )
            mlflow.log_dict(params, "config/effective.json")
            for filename in (
                "MLproject",
                "python_env.yaml",
                "requirements.txt",
                "mibici/config.py",
            ):
                mlflow.log_artifact(str(config.PROJ_ROOT / filename), "config")
            if hasattr(model, "feature_importances_"):
                pd.DataFrame(
                    {"feature": features, "importance": model.feature_importances_}
                ).to_csv(model_dir / "feature_importance.csv", index=False)
            for artifact in model_dir.iterdir():
                if artifact.is_file():
                    mlflow.log_artifact(str(artifact), "evaluation")
            mlflow.log_artifacts(str(model_dir / "plots"), "plots")
            mlflow.log_dict(metadata, "evaluation/split_metadata.json")
            mlflow.log_dict(manifest, "data/data_manifest.json")
            mlflow.log_dict(geography, "data/geography_manifest.json")
            mlflow.log_artifact(
                str(config.PROJ_ROOT / "mibici" / "modeling" / "p2.py"), "source"
            )
            mlflow.log_artifact(
                str(config.PROJ_ROOT / "mibici" / "p2_panel.py"), "source"
            )
            mlflow.log_artifact(str(config.PROJ_ROOT / "requirements.txt"), "source")
            # Solo se autoriza el árbol del estimador recién entrenado aquí;
            # no se cargan archivos externos ni listas automáticas de tipos.
            # Guardar firma y ejemplo; comprobar después la recuperación del modelo.
            example = X_train.head(3)
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=example,
                signature=infer_signature(example, model.predict(example)),
                pip_requirements=str(config.PROJ_ROOT / "requirements.txt"),
                skops_trusted_types=(
                    ["sklearn.tree._tree.Tree"] if family == "random_forest" else []
                ),
            )
            restored = mlflow.sklearn.load_model(info.model_uri)
            np.testing.assert_allclose(
                restored.predict(example), model.predict(example)
            )
            mlflow.set_tag("model_reload_verified", "true")
            comparisons.append(
                {
                    "model": name,
                    "model_family": family,
                    "run_id": run.info.run_id,
                    "model_uri": info.model_uri,
                    "hyperparameters": json.dumps(
                        model.get_params(deep=True), default=str, sort_keys=True
                    ),
                    **metrics,
                }
            )
    # Elegir con validación exclusivamente; prueba solo informa desempeño final.
    selected_row = max(comparisons, key=lambda row: row["validation_f1"])
    selected = selected_row["model"]
    baseline = next(
        (row for row in comparisons if row["model_family"] == "dummy_most_frequent"),
        None,
    )
    if baseline is not None:
        for row in comparisons:
            row["validation_f1_improvement"] = (
                row["validation_f1"] - baseline["validation_f1"]
            )
            row["validation_f1_improvement_percent"] = (
                100 * row["validation_f1_improvement"] / baseline["validation_f1"]
                if baseline["validation_f1"]
                else None
            )
    results = {
        "task": "P2",
        "target_definition": TARGET,
        "limitation": "No observed stockouts; fixed reference capacity from September 2026",
        "experiment": config.P2_EXPERIMENT,
        "parent_run_id": (
            mlflow.active_run().info.run_id if mlflow.active_run() else None
        ),
        "git_commit": commit,
        "data_version": manifest["data_version"],
        "selected_by_validation_f1": selected,
        "split_metadata": metadata,
        "models": comparisons,
    }
    pd.DataFrame(comparisons).to_csv(report_dir / "comparison.csv", index=False)
    (report_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    if mlflow.active_run():
        mlflow.set_tag("selected_model", selected)
        mlflow.set_tag("selected_run_id", selected_row["run_id"])
        mlflow.log_artifact(str(report_dir / "comparison.csv"), "comparison")
        mlflow.log_artifact(str(report_dir / "results.json"), "comparison")
    print(
        pd.DataFrame(comparisons)[["model", "validation_f1", "test_f1"]].to_string(
            index=False
        )
    )
    print(f"Modelo elegido por F1 de validación: {selected}")
    print(f"Resultados: {report_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=config.P2_DATA_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--report-dir", type=Path, default=config.P2_REPORT_DIR)
    parser.add_argument(
        "--boundary-manifest", type=Path, default=config.P1_BOUNDARY_MANIFEST
    )
    args = parser.parse_args()
    run_p2(
        args.manifest,
        args.data_dir,
        args.tracking_uri,
        args.report_dir,
        args.boundary_manifest,
    )


if __name__ == "__main__":
    main()
