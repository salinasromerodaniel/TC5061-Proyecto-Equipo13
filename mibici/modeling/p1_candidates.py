"""Experimentación P1: búsqueda temporal y artefactos para el Model Registry."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from mibici import config
from mibici.dataset import DataError, read_manifest, verify_manifest
from mibici.features import p1_station_week_panel, weekly_departures
from mibici.geography import load_boundary, municipality_station_ids
from mibici.modeling.p1 import baseline_predictions, regression_metrics


def candidate_features(panel):
    """Retardos ya desplazados y calendario conocido al emitir el pronóstico."""
    features = panel[["lag_1", "mean_previous_4", "lag_52"]].astype(float).copy()
    # Calendario de la semana objetivo, no una observación del futuro.
    phase = 2 * np.pi * (panel.week_start.dt.dayofyear - 1) / 365.2425
    features["week_sin"] = np.sin(phase)
    features["week_cos"] = np.cos(phase)
    return features[config.P1_CANDIDATE_FEATURES]


def relative_improvement(reference, value):
    """Sin porcentaje definido cuando el error de referencia ya es cero."""
    return None if reference == 0 else 100 * (reference - value) / reference


def candidate_specs(model_name, parameters):
    if model_name not in ("hgb-search", "hgb", "baseline"):
        raise DataError("P1 admite hgb-search, hgb o baseline.")
    if model_name == "baseline":
        if parameters:
            raise DataError("El baseline previous_week no tiene hiperparámetros ajustables.")
        return []
    specs = config.P1_HGB_SEARCH if model_name == "hgb-search" else [{}]
    result = []
    for index, spec in enumerate(specs, 1):
        actual = {**config.P1_HGB_DEFAULTS, **parameters, **spec}
        if actual["loss"] != "poisson" or actual["early_stopping"] is not False:
            raise DataError("P1 usa pérdida Poisson y entrenamiento sin validación aleatoria interna.")
        estimator = HistGradientBoostingRegressor(**actual)
        result.append((f"hgb_{index}_leaves{actual['max_leaf_nodes']}", estimator))
    return result


def performance_figures(frame, destination, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("actual_vs_predicted", "residuals"):
        fig, ax = plt.subplots(figsize=(7, 4))
        if suffix == "actual_vs_predicted":
            ax.scatter(frame.departures, frame.prediction, alpha=.35, s=12)
            end = float(max(frame.departures.max(), frame.prediction.max()))
            ax.plot([0, end], [0, end], linestyle="--", color="black")
            ax.set(xlabel="Viajes observados", ylabel="Viajes pronosticados")
        else:
            ax.scatter(frame.prediction, frame.departures - frame.prediction, alpha=.35, s=12)
            ax.axhline(0, linestyle="--", color="black")
            ax.set(xlabel="Viajes pronosticados", ylabel="Residual: observado − pronosticado")
        ax.set_title(f"P1 · prueba · {name}")
        ax.grid(alpha=.2)
        fig.tight_layout()
        path = destination / f"{name}_{suffix}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def prepare_panel(manifest_path, data_dir):
    manifest = read_manifest(manifest_path)
    verify_manifest(manifest, data_dir)
    geometry, geography = load_boundary(data_dir)
    station_ids = municipality_station_ids(
        pd.read_parquet(Path(data_dir) / "processed/stations.parquet"), geometry)
    counts = weekly_departures(manifest, data_dir)
    panel, split = p1_station_week_panel(
        counts, station_ids, manifest["months"], cohort_start_month=config.P1_COHORT_START_MONTH)
    return panel, manifest, split, geography


def run_candidates(model_name, parameters, manifest_path, data_dir, report_dir, parent_id, commit):
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    # Validar especificaciones antes de leer los datos o crear runs hijos.
    specs = candidate_specs(model_name, parameters)
    panel, manifest, split, geography = prepare_panel(manifest_path, data_dir)
    X = candidate_features(panel)
    masks = {part: panel.split.eq(part) for part in ("train", "validation", "test")}
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    records = []
    fitted = {}
    for name, estimator in [("previous_week", None), *specs]:
        family = "persistence" if estimator is None else "hist_gradient_boosting_poisson"
        stage = "baseline" if estimator is None else "candidate"
        resolved = {} if estimator is None else estimator.get_params()
        with mlflow.start_run(run_name=f"P1_{name}", nested=True,
                              tags={"team": "13", "question": "P1", "model_family": family,
                                    "stage_of_work": stage, "git_commit": commit,
                                    "mlflow.source.git.commit": commit}) as run:
            mlflow.log_params({**resolved, "model": name, "seed": resolved.get("random_state", config.RANDOM_SEED),
                               "data_version": manifest["data_version"],
                               "data_start_month": manifest["months"][0],
                               "data_end_month": manifest["months"][-1],
                               "features": ",".join(X.columns),
                               "validation_start": split["validation_start"],
                               "test_start": split["test_start"],
                               "test_end_exclusive": split["test_end_exclusive"],
                               "cohort_stations": split["cohort_stations"],
                               "horizon_days": split["horizon_days"],
                               "evaluation_protocol": split["evaluation_protocol"]})
            mlflow.log_dict(manifest, "data/data_manifest.json")
            mlflow.log_dict(geography, "data/geography_manifest.json")
            mlflow.log_dict(split, "data/split_metadata.json")
            mlflow.log_dict({"model": name, "parameters": resolved, "features": list(X.columns),
                             "seed": config.RANDOM_SEED, "selection": "validation_mae_only",
                             "early_stopping": False,
                             "training_rows": int(masks["train"].sum())}, "config/run_config.json")
            for filename in ("MLproject", "python_env.yaml", "requirements.txt", "mibici/config.py"):
                mlflow.log_artifact(str(config.PROJ_ROOT / filename), "config")
            training = X.loc[masks["train"]].copy()
            training["departures"] = panel.loc[masks["train"], "departures"].to_numpy()
            mlflow.log_input(mlflow.data.from_pandas(training,
                source=Path(manifest_path).resolve().as_uri(), targets="departures",
                name="p1_station_week_train"), context="training")
            start = perf_counter()
            if estimator is not None:
                estimator.fit(X.loc[masks["train"]], panel.loc[masks["train"], "departures"])
            fit_seconds = perf_counter() - start
            predictions = (baseline_predictions(panel, "previous_week") if estimator is None
                           else estimator.predict(X))
            metrics = {f"{part}_{key}": value for part in ("train", "validation")
                       for key, value in regression_metrics(panel.loc[masks[part], "departures"],
                                                           predictions[masks[part]]).items()}
            mlflow.log_metrics({**metrics, "fit_seconds": fit_seconds})
            example = X.loc[masks["train"]].head(3)
            output_example = example.lag_1.to_numpy() if estimator is None else estimator.predict(example)
            signature = infer_signature(example, output_example)
            if estimator is None:
                info = mlflow.pyfunc.log_model(name="model",
                    python_model=str(config.PROJ_ROOT / "mibici/modeling/persistence_model.py"),
                    signature=signature, input_example=example,
                    pip_requirements=str(config.PROJ_ROOT / "requirements.txt"))
            else:
                info = mlflow.sklearn.log_model(estimator, name="model", signature=signature,
                    input_example=example, pip_requirements=str(config.PROJ_ROOT / "requirements.txt"),
                    skops_trusted_types=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"])
            record = {"model": name, "model_family": family, "stage_of_work": stage,
                      "run_id": run.info.run_id, "model_uri": info.model_uri,
                      "parameters": resolved, "fit_seconds": fit_seconds, **metrics}
            records.append(record)
            fitted[name] = predictions
    # Congelar elección antes de calcular o registrar las métricas de prueba.
    selected = min(records, key=lambda item: item["validation_mae"])
    mlflow.log_params({"selected_model": selected["model"], "selected_run_id": selected["run_id"],
                       "selection_metric": "validation_mae", "data_version": manifest["data_version"]})
    mlflow.log_dict({"chosen_before_test": True, "run_id": selected["run_id"],
                     "validation_mae": selected["validation_mae"]}, "selection/validation_choice.json")
    for record in records:
        name = record["model"]
        prediction = fitted[name]
        with mlflow.start_run(run_id=record["run_id"], nested=True):
            start = perf_counter()
            loaded = mlflow.pyfunc.load_model(record["model_uri"])
            reloaded = np.asarray(loaded.predict(X.loc[masks["test"]]), dtype=float)
            inference_seconds = perf_counter() - start
            np.testing.assert_allclose(reloaded, prediction[masks["test"]], rtol=1e-12, atol=1e-12)
            test_metrics = {f"test_{key}": value for key, value in regression_metrics(
                panel.loc[masks["test"], "departures"], reloaded).items()}
            mlflow.log_metrics({**test_metrics, "load_and_batch_predict_seconds": inference_seconds})
            record.update(test_metrics)
            record["load_and_batch_predict_seconds"] = inference_seconds
            record["test_mae_improvement_pct"] = relative_improvement(records[0]["test_mae"], record["test_mae"])
            record["validation_mae_improvement_pct"] = relative_improvement(records[0]["validation_mae"], record["validation_mae"])
            frame = panel.loc[masks["test"], ["station_id", "week_start", "departures"]].copy()
            frame["prediction"] = reloaded
            for figure in performance_figures(frame, report_dir / "plots", name):
                mlflow.log_artifact(str(figure), "plots")
            mlflow.log_dict({"weekly_mae": frame.assign(error=(frame.departures-frame.prediction).abs())
                            .groupby("week_start").error.mean().rename_axis("week_start")
                            .rename(index=lambda t: t.date().isoformat()).to_dict()},
                           "evaluation/weekly_stability.json")
            mlflow.set_tag("selected_by_validation", str(record["run_id"] == selected["run_id"]).lower())
    handoff = {"question": "P1", "experiment": config.PROJECT_EXPERIMENT,
               "parent_run_id": parent_id, "selected_run_id": selected["run_id"],
               "selected_model": selected["model"], "model_uri": selected["model_uri"],
               "git_commit": commit, "data_version": manifest["data_version"],
               "suggested_registered_name": "mibici-equipo13-p1-demanda",
               "selection_metric": "validation_mae", "registry_status": "pending_part_c",
               "features": list(X.columns)}
    result = {"question": "P1", "experiment": config.PROJECT_EXPERIMENT,
              "git_commit": commit, "data_version": manifest["data_version"],
              "parent_run_id": parent_id, "split": split, "models": records,
              "selected_model": selected["model"], "handoff": handoff}
    mlflow.log_dict(handoff, "selection/registry_handoff.json")
    mlflow.log_dict(result, "evaluation/results.json")
    (report_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (report_dir / "registry_handoff.json").write_text(json.dumps(handoff, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([{k:v for k,v in rec.items() if k != "parameters"} for rec in records]).to_csv(
        report_dir / "comparison.csv", index=False)
    return result
