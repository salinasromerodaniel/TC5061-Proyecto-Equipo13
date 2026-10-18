import json
import mlflow
import matplotlib
import matplotlib.pyplot as plt

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

def run_p3(manifest_path=config.P1_DATA_MANIFEST,
           data_dir=config.DATA_DIR):

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

    model = KMeans(
        n_clusters=5,
        random_state=config.RANDOM_SEED,
        n_init=10,
    )

    labels = model.fit_predict(X_scaled)

    features["cluster"] = labels

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

    report_dir = config.REPORTS_DIR / "p3"
        
    report_dir.mkdir(
            parents=True,
            exist_ok=True,
            )

    uri = (
    "sqlite:///" +
    str(config.P1_TRACKING_DB)
    )

    mlflow.set_tracking_uri(uri)

    mlflow.set_experiment(
    config.P1_EXPERIMENT
    )


    with mlflow.start_run(
        run_name="P3_kmeans_baseline",
        tags={
            "team": "13",
            "question": "P3",
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
            5,
        )

        mlflow.log_param(
            "data_version",
            manifest["data_version"],
        )

        mlflow.log_metrics(metrics)

        mlflow.log_artifact(
            str(artifact),
            artifact_path="figures",
        )

        mlflow.log_artifact(
            str(clusters_path),
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

if __name__ == "__main__":
    print(
        json.dumps(
            run_p3(),
            indent=2,
            ensure_ascii=False,
        )
    )
