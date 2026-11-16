"""Entrenamiento del retador y comparación offline de versiones registradas de P1."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from mibici import config
from mibici.dataset import DataError
from mibici.modeling.p1 import regression_metrics
from mibici.modeling.p1_candidates import candidate_specs
from mibici.monitoring import DEFAULT_DATA, dump_json, load_partitions
from mibici.serving import LoadedP1, registry_connection, validate_signature

PUBLIC_CONTRACT = config.PROJ_ROOT / "reports/monitoring/p1/data-contract.json"
DELIVERY4_BASELINE = config.PROJ_ROOT / "references/p1-ab-baseline-delivery4.json"
EXPERIMENT = "mibici-equipo13-p1-ab"
DEFAULT_OUTPUT = config.DATA_DIR / "monitoring/ab"


def ab_experiment_id():
    """Resolver el experimento sin cambiar el entorno de los procesos hijos."""
    import mlflow
    from mlflow.exceptions import MlflowException

    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is not None:
        return experiment.experiment_id
    try:
        return mlflow.create_experiment(EXPERIMENT)
    except MlflowException as error:
        if error.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
        return mlflow.get_experiment_by_name(EXPERIMENT).experiment_id


def load_frozen_data(data_dir):
    parts, contract = load_partitions(data_dir)
    frozen = json.loads(PUBLIC_CONTRACT.read_text(encoding="utf-8"))
    if contract["contract_id"] != frozen["contract_id"]:
        raise DataError("Las particiones no corresponden al contrato publicado de P1.")
    return parts, contract


def load_registered(client, name, version, contract, *, is_a, baseline=None):
    import mlflow

    registered = client.get_model_version(name, version)
    if registered.status != "READY" or not registered.run_id:
        raise DataError(f"La versión {version} no está lista o no tiene corrida.")
    run = client.get_run(registered.run_id)
    uri = f"models:/{name}/{version}"
    identity = {"name": name, "version": version, "registered_uri": uri,
                "run_id": registered.run_id, "data_version": run.data.params.get("data_version"),
                "git_commit": run.data.tags.get("git_commit", run.data.tags.get("mlflow.source.git.commit"))}
    if is_a:
        expected = baseline if baseline is not None else contract["model_a"]
        if any(identity[key] != expected[key] for key in identity):
            raise DataError("La versión A del Registry no coincide con el linaje fijado.")
    elif (run.data.params.get("contract_id") != contract["contract_id"]
          or run.data.tags.get("stage_of_work") != "p1_ab_challenger"
          or run.info.status != "FINISHED"):
        raise DataError("La versión B no procede del entrenamiento alternativo de este contrato.")

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=tmp))
        sha = hashlib.sha256((directory / "model.skops").read_bytes()).hexdigest()
        if is_a and sha != expected["model_sha256"]:
            raise DataError("El archivo del modelo A no coincide con el SHA-256 fijado.")
        identity["model_sha256"] = sha
        # El modelo se carga mientras su directorio existe. Para P1, skops carga
        # el estimador completo en memoria.
        model = mlflow.pyfunc.load_model(str(directory))
        validate_signature(model)
    return LoadedP1(model, identity)


def verify_reproduced_baseline(parts, contract, model_a):
    """Confirmar que el A local reproduce P1 usando solo las particiones históricas."""
    features, target = contract["features"], contract["target"]
    ref = parts["reference"]
    _, estimator = candidate_specs("hgb", {"max_leaf_nodes": 7})[0]
    results = {}
    with threadpool_limits(limits=1):
        estimator.fit(ref[features], ref[target])
        for name in ("reference", "historical_validation", "historical_test"):
            frame = parts[name]
            X = frame[features]
            original = model_a.predict(X)
            rerun = estimator.predict(X)
            if not np.allclose(original, rerun, rtol=0, atol=1e-10):
                raise DataError(f"El modelo A local no reproduce el histórico en {name}.")
            results[name] = {"rows": len(frame),
                             "max_abs_prediction_difference": float(np.max(np.abs(original - rerun)))}
    return results


def log_challenger(reference, validation, contract):
    """Entrenar, seleccionar y guardar B en el tracking sin registrarlo aún."""
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    features, target = contract["features"], contract["target"]
    X_train, y_train = reference[features], reference[target]
    X_val, y_val = validation[features], validation[target]
    trials = []
    # Las siete hojas corresponden a A. Los candidatos comparten datos y familia.
    with threadpool_limits(limits=1):
        for leaves in (15, 31):
            _, estimator = candidate_specs("hgb", {"max_leaf_nodes": leaves})[0]
            estimator.fit(X_train, y_train)
            metrics = regression_metrics(y_val, estimator.predict(X_val))
            trials.append((estimator, metrics))
        selected, scores = min(trials, key=lambda trial: trial[1]["mae"])
        example = X_train.head(3)
        signature = infer_signature(example, selected.predict(example))

    with mlflow.start_run(experiment_id=ab_experiment_id(), run_name="P1_entregable8_B", tags={
        "question": "P1", "stage_of_work": "p1_ab_challenger", "selected_by_validation": "true",
    }):
        mlflow.log_params({**selected.get_params(), "contract_id": contract["contract_id"],
                           "data_version": contract["historical_data_version"],
                           "training_rows": len(reference), "validation_rows": len(validation)})
        mlflow.log_metrics({f"validation_{key}": value for key, value in scores.items()})
        mlflow.log_dict([
            {"max_leaf_nodes": estimator.max_leaf_nodes, **metrics} for estimator, metrics in trials
        ], "selection/validation.json")
        mlflow.log_dict(contract, "data/contract.json")
        mlflow.log_artifact(__file__, "source")
        return mlflow.sklearn.log_model(
            selected, name="model", signature=signature, input_example=example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
            pip_requirements=str(config.PROJ_ROOT / "requirements.txt"),
            skops_trusted_types=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],
        )


def train_challenger(reference, validation, contract):
    import mlflow

    info = log_challenger(reference, validation, contract)
    return mlflow.register_model(info.model_uri, contract["model_a"]["name"], tags={
        "question": "P1", "role": "challenger", "contract_id": contract["contract_id"],
    })


def evaluate(evaluation, contract, model_a, model_b):
    features, target = contract["features"], contract["target"]
    X = evaluation[features]
    with threadpool_limits(limits=1):
        predictions = {"A": model_a.predict(X), "B": model_b.predict(X)}
    scores = {arm: regression_metrics(evaluation[target], pred)
              for arm, pred in predictions.items()}
    weekly = []
    for week, group in evaluation.assign(**predictions).groupby("week_start", sort=True):
        mae_a = regression_metrics(group[target], group.A)["mae"]
        mae_b = regression_metrics(group[target], group.B)["mae"]
        weekly.append({"week_start": week.date().isoformat(), "rows": len(group),
                       "mae_a": mae_a, "mae_b": mae_b, "delta_mae_b_minus_a": mae_b - mae_a})
    mae_a, mae_b = scores["A"]["mae"], scores["B"]["mae"]
    supports_b = (mae_b < mae_a and all(row["delta_mae_b_minus_a"] < 0 for row in weekly)
                  and scores["B"]["rmse"] <= scores["A"]["rmse"])
    keys = evaluation[contract["keys"]].to_csv(index=False, date_format="%Y-%m-%d",
                                               lineterminator="\n")
    return {
        "contract_id": contract["contract_id"],
        "evaluation": {**contract["partitions"]["current"], "ordered_keys": contract["keys"],
                       "ordered_keys_sha256": hashlib.sha256(keys.encode("utf-8")).hexdigest()},
        "models": {"A": model_a.identity, "B": model_b.identity},
        "metrics": scores,
        "delta_mae_b_minus_a": mae_b - mae_a,
        "relative_mae_improvement_pct": 100 * (mae_a - mae_b) / mae_a if mae_a else None,
        "recommended_version": (model_b if supports_b else model_a).identity["version"],
        "decision_rule": "B si reduce MAE global y en cada semana, sin aumentar RMSE; si no, A.",
        "limitation": f"Comparación descriptiva offline de {len(weekly)} semanas. "
                      "Las observaciones de una misma semana y estación pueden estar correlacionadas.",
        "weekly": weekly,
    }


def save_evaluation(result, output):
    import mlflow

    with mlflow.start_run(experiment_id=ab_experiment_id(), run_name="P1_entregable8_AB",
                          tags={"question": "P1"}):
        result["evaluation_run_id"] = mlflow.active_run().info.run_id
        mlflow.log_params({"contract_id": result["contract_id"],
                           "evaluation_rows": result["evaluation"]["rows"],
                           "recommended_version": result["recommended_version"],
                           **{f"version_{arm.lower()}": item["version"]
                              for arm, item in result["models"].items()}})
        mlflow.log_metrics({f"{arm.lower()}_{key}": value
                            for arm, metrics in result["metrics"].items()
                            for key, value in metrics.items()})
        dump_json(result, output / "results.json")
        pd.DataFrame([
            {"arm": arm, "version": result["models"][arm]["version"], **metrics}
            for arm, metrics in result["metrics"].items()
        ]).to_csv(output / "comparison.csv", index=False)
        pd.DataFrame(result["weekly"]).to_csv(output / "weekly.csv", index=False)
        for name in ("results.json", "comparison.csv", "weekly.csv"):
            mlflow.log_artifact(str(output / name), "evaluation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"),
                        help="Registry que contiene el modelo A fijado.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--baseline", choices=("original", "delivery4-local"), default="original",
                        help="Corrida original acordada o reproducción de la entrega con Docker.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train")
    compare = commands.add_parser("compare")
    compare.add_argument("--version-b", required=True)
    compare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not args.tracking_uri:
        parser.error("Configura MLFLOW_TRACKING_URI o indica --tracking-uri.")
    parts, contract = load_frozen_data(args.data_dir)
    use_local = args.baseline == "delivery4-local"
    model_a = (json.loads(DELIVERY4_BASELINE.read_text(encoding="utf-8"))
               if use_local else contract["model_a"])
    name, version = model_a["name"], model_a["version"]
    if args.command == "compare" and (not args.version_b.isascii() or not args.version_b.isdecimal()
                                       or int(args.version_b) < 1
                                       or str(int(args.version_b)) != args.version_b
                                       or args.version_b == version):
        parser.error("--version-b debe ser una versión registrada distinta de A.")
    with registry_connection(args.tracking_uri) as client:
        a = load_registered(client, name, version, contract, is_a=True, baseline=model_a)
        equivalence = verify_reproduced_baseline(parts, contract, a) if use_local else None
        if args.command == "train":
            b = train_challenger(parts["reference"], parts["historical_validation"], contract)
            print(f"B registrado: models:/{name}/{b.version} (run {b.run_id})")
        else:
            b = load_registered(client, name, args.version_b, contract, is_a=False)
            result = evaluate(parts["current"], contract, a, b)
            if use_local:
                result["baseline_provenance"] = {
                    "kind": "delivery4-local-reproduction",
                    "original_contract_a": contract["model_a"],
                    "historical_prediction_check": equivalence,
                }
            save_evaluation(result, args.output)
            print(f"Versión recomendada: {result['recommended_version']}. Evidencia: {args.output}")


if __name__ == "__main__":
    main()
