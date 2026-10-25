"""P2: déficit neto semanal indirecto; nunca stockouts observados.

Ejecutar desde un commit: python -m mibici.modeling.p2
"""

import argparse
import json
import os
from pathlib import Path
import platform

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
                columns=["started_at", "ended_at", "origin_station_id", "destination_station_id"]):
            trips = batch.to_pandas()
            for station, time, name in (("origin_station_id", "started_at", "departures"),
                                         ("destination_station_id", "ended_at", "arrivals")):
                frame = trips[[station, time]].dropna().copy()
                frame["week_start"] = (frame[time].dt.normalize() -
                                       pd.to_timedelta(frame[time].dt.dayofweek, unit="D"))
                frame = frame.rename(columns={station: "station_id"})
                parts.append(frame.groupby(["station_id", "week_start"]).size().rename(name))
    if not parts:
        raise ValueError("El manifiesto no contiene viajes.")
    # Concatenar por filas conserva el nombre de cada medida como columna.
    return (pd.concat([part.to_frame() for part in parts])
            .groupby(level=[0, 1]).sum().reset_index())


def classification_metrics(actual, predictions, scores):
    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                                 precision_score, recall_score)

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


def confusion_figure(actual, predictions, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(actual, predictions, labels=[0, 1],
                                            display_labels=["Sin proxy", "Con proxy"], ax=ax,
                                            colorbar=False)
    ax.set(title=title, xlabel="Predicción", ylabel="Etiqueta indirecta observada")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_p2(manifest_path=config.P2_DATA_MANIFEST, data_dir=config.DATA_DIR,
           tracking_uri=None, report_dir=config.P2_REPORT_DIR,
           boundary_manifest=config.P1_BOUNDARY_MANIFEST, model_name=None, model_parameters=None):
    """Verificar datos, ajustar solo con train y registrar dos runs reproducibles."""
    import mlflow
    import mlflow.sklearn
    import sklearn
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix

    commit = code_commit()
    manifest = read_manifest(manifest_path)
    verify_manifest(manifest, data_dir)
    counts = weekly_counts(manifest, data_dir)
    stations = pd.read_parquet(Path(data_dir) / "processed" / "stations.parquet")
    geometry, geography = load_boundary(data_dir, boundary_manifest)
    zapopan_ids = municipality_station_ids(stations, geometry)
    if not zapopan_ids:
        raise ValueError("El polígono de Zapopan no contiene estaciones del catálogo.")
    # Filtrar estaciones después de contar ambos extremos de todos los viajes:
    # un viaje que cruza el municipio sí aporta al extremo situado en Zapopan.
    panel, features, metadata = build_panel(
        counts, stations, full_weeks(manifest["months"]),
        config.P2_VALIDATION_WEEKS, config.P2_TEST_WEEKS, station_ids=zapopan_ids)
    metadata.update({"scope": "inegi_zapopan_reference_polygon_historical_cohort",
                     "municipality_code": geography["municipality_code"],
                     "municipality_name": geography["municipality_name"],
                     "boundary_sha256": geography["sha256"],
                     "geographic_station_ids": zapopan_ids,
                     "cohort_station_ids": sorted(int(i) for i in panel.station_id.unique())})
    metadata["time_policy"] = manifest["time_policy"]
    # Auditar eventos excluidos por catálogo/capacidad; no inventar capacidad.
    valid_ids = set(stations.loc[stations.capacity > 0, "station_id"])
    excluded = counts[~counts.station_id.isin(valid_ids)]
    metadata["events_excluded_by_catalog_or_capacity"] = {
        name: int(excluded[name].sum()) for name in ("departures", "arrivals")}
    outside = counts[counts.station_id.isin(stations.station_id) &
                     ~counts.station_id.isin(zapopan_ids)]
    metadata["events_outside_zapopan_reference"] = {
        name: int(outside[name].sum()) for name in ("departures", "arrivals")}
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    train = panel[panel.split == "train"]
    if train.target.nunique() != 2:
        raise ValueError("Entrenamiento necesita ambas clases; revisar etiqueta y periodo.")
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or (
        "sqlite:///" + str(config.P2_TRACKING_DB.resolve()))
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(config.P2_EXPERIMENT) is None:
        mlflow.create_experiment(config.P2_EXPERIMENT,
                                artifact_location=config.P2_ARTIFACT_DIR.resolve().as_uri())
    mlflow.set_experiment(config.P2_EXPERIMENT)
    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "random_forest": RandomForestClassifier(**config.P2_MODEL_DEFAULTS),
    }
    if model_name not in (None, "baseline", "dummy_most_frequent", "random_forest"):
        raise ValueError("P2 admite baseline, dummy_most_frequent o random_forest.")
    if model_name == "baseline":
        models = {"dummy_most_frequent": models["dummy_most_frequent"]}
    elif model_name is not None:
        models = {model_name: models[model_name]}
    if model_parameters:
        if len(models) != 1:
            raise ValueError("Selecciona un modelo P2 antes de indicar hiperparámetros.")
        next(iter(models.values())).set_params(**model_parameters)
    comparisons = []
    for name, model in models.items():
        model_dir = report_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        params = {
            **model.get_params(), "model": name, "seed": config.RANDOM_SEED,
            "data_start_month": manifest["months"][0],
            "data_end_month": manifest["months"][-1],
            "data_version": manifest["data_version"], "horizon_days": 7,
            "history_weeks": 4, "decision_rule": "estimator_predict_default_no_tuning",
            "target_definition": TARGET, "features": ",".join(features),
            "capacity_policy": metadata["capacity_policy"], "scope": metadata["scope"],
            "cohort_stations": metadata["cohort_stations"],
            "municipality_code": geography["municipality_code"],
            "municipality_name": geography["municipality_name"],
            "boundary_sha256": geography["sha256"],
            "geographic_catalog_stations": len(zapopan_ids),
            "python_version": platform.python_version(), "pandas_version": pd.__version__,
            "sklearn_version": sklearn.__version__, "mlflow_version": mlflow.__version__,
        }
        for split, details in metadata["partitions"].items():
            params.update({f"{split}_{key}": value for key, value in details.items()})
        with mlflow.start_run(
                run_name=f"P2_Zapopan_{name}_{manifest['months'][0]}_{manifest['months'][-1]}",
                nested=mlflow.active_run() is not None,
                tags={"team": "13", "question": "P2", "git_commit": commit,
                      "model_family": name, "stage_of_work": "baseline" if name.startswith("dummy") else "candidate",
                      "mlflow.source.git.commit": commit, "target_kind": "indirect_proxy",
                      "capacity_reference": "2026-09_retrospective",
                      "municipality": "Zapopan"}) as run:
            model.fit(train[features], train.target)
            mlflow.log_params(params)
            metrics = {}
            for split in ("validation", "test"):
                frame = panel[panel.split == split]
                prediction = model.predict(frame[features])
                scores = model.predict_proba(frame[features])[:, list(model.classes_).index(1)]
                metrics.update({f"{split}_{key}": value for key, value in
                                classification_metrics(frame.target, prediction, scores).items()})
                matrix = confusion_matrix(frame.target, prediction, labels=[0, 1])
                pd.DataFrame(matrix, index=["real_0", "real_1"], columns=["pred_0", "pred_1"]).to_csv(
                    model_dir / f"{split}_confusion_matrix.csv")
                mlflow.log_dict(classification_report(frame.target, prediction, labels=[0, 1],
                                                      output_dict=True, zero_division=0),
                                f"evaluation/{split}_classification_report.json")
                confusion_figure(frame.target, prediction, model_dir / f"{split}_confusion.png",
                                 f"P2 {name} · {split}")
                output = frame[["station_id", "week_start", "target_week", "target"]].copy()
                output["prediction"] = prediction
                output["score"] = scores
                output.to_csv(model_dir / f"{split}_predictions.csv", index=False)
            mlflow.log_metrics(metrics)
            if hasattr(model, "feature_importances_"):
                pd.DataFrame({"feature": features, "importance": model.feature_importances_}).to_csv(
                    model_dir / "feature_importance.csv", index=False)
            mlflow.log_artifacts(str(model_dir), artifact_path="evaluation")
            mlflow.log_dict(metadata, "evaluation/split_metadata.json")
            mlflow.log_dict(manifest, "data/data_manifest.json")
            mlflow.log_dict(geography, "data/geography_manifest.json")
            mlflow.log_artifact(str(config.PROJ_ROOT / "mibici" / "modeling" / "p2.py"), "source")
            mlflow.log_artifact(str(config.PROJ_ROOT / "mibici" / "p2_panel.py"), "source")
            mlflow.log_artifact(str(config.PROJ_ROOT / "requirements.txt"), "source")
            # Solo se autoriza el árbol del estimador recién entrenado aquí;
            # no se cargan archivos externos ni listas automáticas de tipos.
            mlflow.sklearn.log_model(
                model, name="model", input_example=train[features].head(2),
                skops_trusted_types=(
                    ["sklearn.tree._tree.Tree"] if name == "random_forest" else []),
            )
            comparisons.append({"model": name, "run_id": run.info.run_id, **metrics})
    # Elegir con validación exclusivamente; prueba solo informa desempeño final.
    selected = max(comparisons, key=lambda row: row["validation_f1"])["model"]
    results = {"task": "P2", "target_definition": TARGET,
               "limitation": "No observed stockouts; fixed reference capacity from September 2026",
               "git_commit": commit, "data_version": manifest["data_version"],
               "selected_by_validation_f1": selected, "split_metadata": metadata,
               "models": comparisons}
    pd.DataFrame(comparisons).to_csv(report_dir / "comparison.csv", index=False)
    (report_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(f"Modelo elegido por F1 de validación: {selected}")
    print(f"Resultados: {report_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=config.P2_DATA_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--report-dir", type=Path, default=config.P2_REPORT_DIR)
    parser.add_argument("--boundary-manifest", type=Path, default=config.P1_BOUNDARY_MANIFEST)
    args = parser.parse_args()
    run_p2(args.manifest, args.data_dir, args.tracking_uri, args.report_dir,
           args.boundary_manifest)


if __name__ == "__main__":
    main()
