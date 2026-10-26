import json
from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import matplotlib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from mibici import config
from mibici.dataset import (
    read_manifest,
    verify_manifest,
)

from mibici.features import p3_station_features
from mibici.modeling.p1 import code_commit

import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(config.DATA_DIR / "cache" / "matplotlib"),
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run_p3(manifest_path=config.P1_DATA_MANIFEST,
           data_dir=config.DATA_DIR, report_dir=None, model_parameters=None, tracking_uri=None):

    manifest = read_manifest(manifest_path)

    verify_manifest(
        manifest,
        data_dir
    )

    commit = code_commit()

    features = p3_station_features(
        manifest,
        data_dir
    )

    report_dir = Path(report_dir) if report_dir is not None else config.REPORTS_DIR / "p3"
            
    report_dir.mkdir(
                parents=True,
                exist_ok=True,
                )
    

    X = features[
        [
            "departures",
            "arrivals",
            "traffic",
            "balance",
            "capacity",
        ]
    ]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    parameters = {**config.P3_MODEL_DEFAULTS, **(model_parameters or {})}
    model = KMeans(**parameters)

    labels = model.fit_predict(X_scaled)

    features["cluster"] = labels

    signature = infer_signature(
    X,
    labels
    )

    clusters_path = report_dir / "clusters.csv"
    features.to_csv(clusters_path, index=False)

    metrics = {
        "silhouette_score":
            float(silhouette_score(X_scaled, labels)),

        "davies_bouldin_score":
            float(davies_bouldin_score(X_scaled, labels)),

        "calinski_harabasz_score":
            float(calinski_harabasz_score(X_scaled, labels)),
    }
    artifact_dir = config.FIGURES_DIR / "p3"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifact = artifact_dir / "p3_clusters.png"

    plt.figure(figsize=(8, 6))

    plt.scatter(
        features["departures"],
        features["arrivals"],
        c=labels,
    )

    plt.xlabel("Departures")
    plt.ylabel("Arrivals")
    plt.title("P3 KMeans Clusters")

    plt.savefig(
        artifact,
        bbox_inches="tight",
        dpi=150,
    )

    plt.close()

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or (
        "sqlite:///" + str(config.P1_TRACKING_DB))

    mlflow.set_tracking_uri(uri)

    mlflow.set_experiment(
    config.P1_EXPERIMENT
    )


    with mlflow.start_run(
        run_name="P3_kmeans_baseline",
        nested=mlflow.active_run() is not None,
        tags={
            "team": "13",
            "question": "P3",
            "model_family": "kmeans",
            "stage_of_work": "baseline",
            "git_commit": commit,
            "mlflow.source.git.commit": commit,
        },
    ):

        mlflow.log_param(
            "algorithm",
            "KMeans",
        )

        mlflow.log_param(
            "n_clusters",
            model.n_clusters,
        )

        mlflow.log_param(
            "data_version",
            manifest["data_version"],
        )

        mlflow.log_params({"seed": parameters["random_state"], "n_init": parameters["n_init"]})
        mlflow.log_dict(manifest, "data/data_manifest.json")
        mlflow.log_dict(parameters, "config/model_parameters.json")
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X.head(5),
        )

        mlflow.log_artifact(
            str(artifact),
            artifact_path="figures",
        )

        mlflow.log_artifact(
            str(clusters_path),
            artifact_path="data",
        )

        mlflow.log_artifact(
            str(manifest_path),
            artifact_path="data",
        )

    result = {
        "question": "P3",
        "data_version": manifest["data_version"],
        "metrics": metrics,
        "clusters": int(features.cluster.nunique()),
        }
    
    (report_dir / "results.json").write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
        )
    return result    

def run_p3_search(
    manifest_path=config.P1_DATA_MANIFEST,
    data_dir=config.DATA_DIR,
    report_dir=None,
):
    import mlflow

    results = []

    search_space = [4, 5, 6, 7]

    with mlflow.start_run(
        run_name="P3_kmeans_search",
        nested=mlflow.active_run() is not None,
    ):

        mlflow.set_tags(
            {
                "question": "P3",
                "model_family": "kmeans",
                "stage_of_work": "search",
            }
        )

        for k in search_space:

            with mlflow.start_run(
                run_name=f"P3_kmeans_k{k}",
                nested=True,
            ):

                result = run_p3(
                    manifest_path=manifest_path,
                    data_dir=data_dir,
                    report_dir=report_dir,
                    model_parameters={
                        "n_clusters": k
                    }
                )

                results.append(
                    {
                        "k": k,
                        **result["metrics"],
                    }
                )

    return results


if __name__ == "__main__":
    print(
        json.dumps(
            run_p3(),
            indent=2,
            ensure_ascii=False,
        )
    )
