"""Baselines reproducibles de demanda semanal por estación de Zapopan (P1)."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from mibici import config
from mibici.dataset import DataError, read_manifest, verify_manifest
from mibici.features import p1_station_week_panel, weekly_departures
from mibici.geography import load_boundary, municipality_station_ids

BASELINES = ("previous_week", "mean_previous_weeks", "seasonal_52_weeks")


def baseline_predictions(panel, name, history_weeks=config.P1_HISTORY_WEEKS):
    """Pronosticar con información anterior al lunes objetivo, sin ajustar al test."""
    if name == "previous_week":
        return panel.lag_1.to_numpy(dtype=float)
    if name == "mean_previous_weeks":
        return panel[f"mean_previous_{history_weeks}"].to_numpy(dtype=float)
    if name == "seasonal_52_weeks":
        return panel.lag_52.fillna(panel.lag_1).to_numpy(dtype=float)
    raise DataError(f"Baseline desconocido: {name}.")


def regression_metrics(actual, predictions):
    """MAE/RMSE en viajes por estación-semana y WAPE cuando hay volumen observado."""
    actual, predictions = np.asarray(actual, dtype=float), np.asarray(predictions, dtype=float)
    if not len(actual) or len(actual) != len(predictions):
        raise DataError("Las métricas requieren observaciones y predicciones de igual tamaño.")
    if not np.isfinite(actual).all() or not np.isfinite(predictions).all():
        raise DataError("No se permiten valores no finitos en evaluación.")
    result = {"mae": float(mean_absolute_error(actual, predictions)),
              "rmse": float(root_mean_squared_error(actual, predictions))}
    if actual.sum() > 0:
        result["wape"] = float(np.abs(actual - predictions).sum() / actual.sum())
    return result


def code_commit():
    """Exigir que el código y dependencias ejecutados correspondan al commit registrado."""
    paths = ["mibici", "requirements.in", "requirements.txt", "pyproject.toml"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.PROJ_ROOT,
                                     text=True).strip()
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *paths],
                      cwd=config.PROJ_ROOT).returncode:
        raise DataError("Guarda el código y dependencias en un commit antes de registrar runs.")
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard",
                                        "--", *paths], cwd=config.PROJ_ROOT, text=True)
    if untracked.strip():
        raise DataError("Hay código sin versionar; inclúyelo en un commit antes del experimento.")
    return commit


def plot_predictions(predictions, baseline_name, figure_dir):
    """Generar artefacto agregado de prueba sin datos de usuario ni viajes individuales."""
    os.environ.setdefault("MPLCONFIGDIR", str(config.DATA_DIR / "cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test = predictions[predictions.split == "test"]
    totals = test.groupby("week_start")[["departures", "prediction"]].sum()
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / f"p1_{baseline_name}_test.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(totals.index, totals.departures, marker="o", label="Observado")
    ax.plot(totals.index, totals.prediction, marker="o", label="Pronóstico")
    ax.set(title=f"P1 · Zapopan · {baseline_name}", xlabel="Semana que inicia el lunes",
           ylabel="Viajes iniciados (cohorte evaluada)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run_p1(manifest_path=config.P1_DATA_MANIFEST, data_dir=config.DATA_DIR,
           boundary_manifest=config.P1_BOUNDARY_MANIFEST, tracking_uri=None,
           experiment_name=config.P1_EXPERIMENT, report_dir=config.P1_REPORT_DIR):
    """Ejecutar y registrar tres baselines; integrable desde el orquestador del equipo."""
    import mlflow

    manifest_path, data_dir, report_dir = Path(manifest_path), Path(data_dir), Path(report_dir)
    commit = code_commit()
    manifest = read_manifest(manifest_path)
    verify_manifest(manifest, data_dir)
    geometry, geography = load_boundary(data_dir, boundary_manifest)
    stations = pd.read_parquet(data_dir / "processed" / "stations.parquet")
    station_ids = municipality_station_ids(stations, geometry)
    counts = weekly_departures(manifest, data_dir)
    panel, split_metadata = p1_station_week_panel(
        counts, station_ids, manifest["months"], cohort_start_month=config.P1_COHORT_START_MONTH)
    unknown = counts[~counts.station_id.isin(stations.station_id)]
    split_metadata["origin_ids_missing_from_catalog"] = sorted(int(i) for i in unknown.station_id.unique())
    split_metadata["starts_missing_origin_coordinates"] = int(unknown.departures.sum())
    split_metadata["scope"] = "origin_coordinates_covered_by_inegi_zapopan_reference_polygon"
    split_metadata["source_time_policy"] = manifest["time_policy"]
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = config.FIGURES_DIR / "p1"
    prediction_dir = data_dir / "processed" / "p1"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or ("sqlite:///" + str(config.P1_TRACKING_DB))
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=config.P1_ARTIFACT_DIR.resolve().as_uri())
    mlflow.set_experiment(experiment_name)
    model_results = []
    for baseline in BASELINES:
        predictions = panel.copy()
        predictions["prediction"] = baseline_predictions(panel, baseline)
        metrics = {}
        for split in ("train", "validation", "test"):
            frame = predictions[predictions.split == split]
            metrics.update({f"{split}_{k}": v for k, v in
                            regression_metrics(frame.departures, frame.prediction).items()})
        if baseline == "seasonal_52_weeks":
            for split in ("train", "validation", "test"):
                metrics[f"{split}_seasonal_fallback_fraction"] = float(
                    panel.loc[panel.split == split, "lag_52"].isna().mean())
        artifact = plot_predictions(predictions, baseline, figure_dir)
        prediction_path = prediction_dir / f"{baseline}_predictions.parquet"
        predictions.to_parquet(prediction_path, index=False)
        params = {"model": baseline, "history_weeks": config.P1_HISTORY_WEEKS,
                  "horizon_days": 7, "forecast_origin": "monday_00_source_time",
                  "seasonal_weeks": config.P1_SEASONAL_WEEKS,
                  "seasonal_fallback": "previous_week",
                  "cohort_observation_start": split_metadata["cohort_observation_start"],
                  "data_start_month": manifest["months"][0],
                  "data_end_month": manifest["months"][-1],
                  "data_version": manifest["data_version"], "seed": config.RANDOM_SEED,
                  "municipality_code": config.P1_MUNICIPALITY_CODE,
                  "boundary_sha256": geography["sha256"],
                  "validation_start": split_metadata["validation_start"],
                  "test_start": split_metadata["test_start"],
                  "test_end_exclusive": split_metadata["test_end_exclusive"],
                  "validation_weeks": config.P1_VALIDATION_WEEKS,
                  "test_weeks": config.P1_TEST_WEEKS,
                  "cohort_stations": split_metadata["cohort_stations"],
                  "evaluation_protocol": split_metadata["evaluation_protocol"],
                  "pandas_version": pd.__version__, "mlflow_version": mlflow.__version__}
        with mlflow.start_run(run_name=f"P1_{baseline}_{manifest['months'][0]}_{manifest['months'][-1]}",
                              tags={"team": "13", "question": "P1", "git_commit": commit,
                                    "mlflow.source.git.commit": commit}) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(artifact), artifact_path="figures")
            mlflow.log_artifact(str(prediction_path), artifact_path="evaluation")
            mlflow.log_dict(split_metadata, "evaluation/split_metadata.json")
            mlflow.log_dict(manifest, "data/data_manifest.json")
            mlflow.log_dict(geography, "data/geography_manifest.json")
            mlflow.log_dict({"target": "observed weekly departures", "primary_metric": "mae",
                             "candidate": "HistGradientBoostingRegressor(loss='poisson')",
                             "candidate_status": "proposed_not_trained",
                             "history_available": "only complete weeks before each forecast origin"},
                            "model/definition.json")
            model_results.append({"model": baseline, "run_id": run.info.run_id, **metrics})
    selected = min(model_results, key=lambda row: row["validation_mae"])["model"]
    result = {"question": "P1", "experiment": experiment_name, "git_commit": commit,
              "data_version": manifest["data_version"], "data_summary": manifest["summary"],
              "geography_sha256": geography["sha256"], "split": split_metadata,
              "selection_metric": "validation_mae", "selected_baseline": selected,
              "models": model_results}
    (report_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    pd.DataFrame(model_results).to_csv(report_dir / "comparison.csv", index=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=config.P1_DATA_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--boundary-manifest", type=Path, default=config.P1_BOUNDARY_MANIFEST)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment", default=config.P1_EXPERIMENT)
    parser.add_argument("--report-dir", type=Path, default=config.P1_REPORT_DIR)
    args = parser.parse_args(argv)
    result = run_p1(args.manifest, args.data_dir, args.boundary_manifest, args.tracking_uri,
                    args.experiment, args.report_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
