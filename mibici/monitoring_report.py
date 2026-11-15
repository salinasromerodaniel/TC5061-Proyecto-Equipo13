"""Reporte Evidently con resultados agregados para el análisis del equipo."""
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np

from mibici.monitoring import FEATURES, dump_json, load_partitions

METHOD = "wasserstein"
THRESHOLD = 0.1


def evaluate(reference, current):
    # Ejecución local: sin envío de telemetría ni publicación en un workspace.
    os.environ.setdefault("DO_NOT_TRACK", "1")
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    definition = DataDefinition(numerical_columns=FEATURES)
    ref = Dataset.from_pandas(reference[FEATURES], data_definition=definition)
    cur = Dataset.from_pandas(current[FEATURES], data_definition=definition)
    return Report([
        DataDriftPreset(columns=FEATURES, method=METHOD, threshold=THRESHOLD, include_tests=False),
        DataSummaryPreset(columns=FEATURES),
    ], include_tests=False).run(current_data=cur, reference_data=ref)


def quality(frame):
    return {
        "rows": len(frame), "duplicate_station_weeks": int(frame.duplicated(["station_id", "week_start"]).sum()),
        "features": {name: {
            "missing_count": int(frame[name].isna().sum()),
            "missing_share": float(frame[name].isna().mean()),
            "infinite_count": int(np.isinf(frame[name].to_numpy(dtype=float)).sum()),
            "min": float(frame[name].min()), "max": float(frame[name].max()),
            "mean": float(frame[name].mean()),
        } for name in FEATURES},
    }


def summarize(snapshot, reference, current, metadata):
    drift = []
    for item in snapshot["metrics"]:
        cfg = item["config"]
        if cfg["type"].endswith(":ValueDrift"):
            score = float(item["value"])
            drift.append({"feature": cfg["column"], "method": cfg["method"],
                          "score": score, "threshold": float(cfg["threshold"]),
                          "drift_detected": score >= float(cfg["threshold"])})
    if {item["feature"] for item in drift} != set(FEATURES):
        raise ValueError("Evidently no devolvió drift para todas las variables.")
    return {"schema_version": 1, "contract_id": metadata["contract_id"],
            "evidently_version": version("evidently"), "features": FEATURES,
            "method": METHOD, "threshold": THRESHOLD,
            "threshold_policy": "exploratory_review_not_retraining_or_promotion_rule",
            "reference": metadata["partitions"]["reference"],
            "current": metadata["partitions"]["current"], "drift": drift,
            "drifted_features": sum(item["drift_detected"] for item in drift),
            "quality": {"reference": quality(reference), "current": quality(current)},
            "limitations": [
                "Current incluye sólo tres semanas y observaciones repetidas de estaciones.",
                "Calendario concentrado en octubre frente a una referencia de varios meses.",
                "lag_52 ausente cuando no existía un año de historia; no se imputa para el reporte.",
                "La cohorte original tiene 78 estaciones: 73 aparecen en train y 78 en current.",
                "Drift de variables no demuestra deterioro del modelo; falta comparación A/B.",
            ]}


def figure(summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = summary["drift"]
    names = [r["feature"] for r in records]
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 13})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    axes[0].barh(names, [r["score"] for r in records], color="#2465a8")
    axes[0].axvline(THRESHOLD, color="#c84a36", linestyle="--", label="Umbral 0.1")
    axes[0].set(xlabel="Wasserstein normalizada", title="Drift · Evidently")
    axes[0].legend(fontsize=12)
    y = np.arange(len(names))
    for offset, part, label in [(-.18, "reference", "Referencia"), (.18, "current", "Actual")]:
        axes[1].barh(y + offset, [100 * summary["quality"][part]["features"][n]["missing_share"]
                               for n in names], height=.36, label=label)
    axes[1].set(yticks=y, yticklabels=names, xlabel="Valores ausentes (%)", title="Calidad de datos")
    axes[1].legend(fontsize=12)
    fig.suptitle("P1 · entrenamiento vs. 5–25 de octubre de 2026")
    fig.tight_layout()
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def create_report(data_dir, report_dir):
    parts, metadata = load_partitions(data_dir)
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    reference, current = parts["reference"], parts["current"]
    result = evaluate(reference, current)
    result.save_html(str(root / "evidently.html"))
    html = root / "evidently.html"
    html.write_text("\n".join(line.rstrip() for line in html.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    result.save_json(str(root / "evidently.json"))
    summary = summarize(result.dict(), reference, current, metadata)
    dump_json(summary, root / "summary.json")
    dump_json(metadata, root / "data-contract.json")
    import pandas as pd
    pd.DataFrame(summary["drift"]).to_csv(root / "drift.csv", index=False)
    figure(summary, root / "drift-quality.png")
    return summary
