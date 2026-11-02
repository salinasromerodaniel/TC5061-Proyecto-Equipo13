"""Validación opcional de P2 con Deepchecks, sin entrenar ni cambiar los datos.

Exportar con el entorno del proyecto; validar el CSV con el entorno Deepchecks.
Las importaciones pesadas son locales para mantener separados ambos entornos.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from mibici import config
from mibici.p2_panel import FEATURES

KEYS = ["station_id", "target_week"]
REQUIRED = ["station_id", "week_start", "target_week", "split", "target", *FEATURES]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding="utf-8")


def panel_errors(panel):
    """Contrato completo antes de entregar datos a Deepchecks; nunca coerce tipos."""
    errors = []
    if panel.columns.duplicated().any():
        return ["Hay nombres de columnas duplicados."]
    missing = sorted(set(REQUIRED) - set(panel.columns))
    extra = sorted(set(panel.columns) - set(REQUIRED))
    if missing:
        errors.append(f"Faltan columnas: {missing}.")
    if extra:
        errors.append(f"Columnas no autorizadas para el reporte: {extra}.")
    if errors:
        return errors
    if panel.empty:
        return ["El panel está vacío."]
    for name in ["station_id", "target", *FEATURES]:
        series = panel[name]
        if not is_numeric_dtype(series) or is_bool_dtype(series):
            errors.append(f"{name} debe tener tipo numérico, no {series.dtype}.")
        elif np.isinf(series.dropna().to_numpy(dtype=float)).any():
            errors.append(f"{name} contiene infinitos.")
    if errors:
        return errors
    # Los nulos numéricos se conservan para que PercentOfNulls los evalúe.
    if not panel.target.dropna().isin([0, 1]).all():
        errors.append("target debe contener exclusivamente 0 y 1.")
    if (panel.capacity.dropna() <= 0).any():
        errors.append("capacity debe ser positiva en el panel de P2.")
    ids = panel.station_id.dropna()
    if ((ids <= 0) | (ids % 1 != 0)).any():
        errors.append("station_id debe contener enteros positivos.")
    for name in ["departures", "arrivals", "departures_mean4", "arrivals_mean4"]:
        if (panel[name].dropna() < 0).any():
            errors.append(f"{name} no puede ser negativo.")
    if not panel["split"].isin(["train", "validation", "test"]).all():
        errors.append("split debe ser train, validation o test, sin nulos.")
    dates = {}
    for name in ["week_start", "target_week"]:
        # Solo se acepta fecha ISO de día completo, no interpretación ambigua.
        dates[name] = pd.to_datetime(panel[name], format="%Y-%m-%d", errors="coerce")
        if dates[name].isna().any() or (dates[name].dt.dayofweek != 0).any():
            errors.append(f"{name} debe contener lunes válidos en formato YYYY-MM-DD.")
    if not errors and not (dates["target_week"] - dates["week_start"] ==
                           pd.Timedelta(days=7)).all():
        errors.append("El objetivo debe estar exactamente una semana después de las features.")
    for split in ["train", "validation", "test"]:
        if (panel["split"] == split).sum() < config.DEEPCHECKS_MIN_ROWS:
            errors.append(f"{split} requiere al menos {config.DEEPCHECKS_MIN_ROWS} filas.")
    if not errors:
        for earlier, later in [("train", "validation"), ("validation", "test")]:
            a = dates["target_week"][panel["split"] == earlier]
            b = dates["target_week"][panel["split"] == later]
            if a.max() >= b.min():
                errors.append(f"Las semanas objetivo de {earlier} y {later} se solapan.")
    return errors


def export_p2(manifest_path, data_dir, output):
    """Reutilizar el pipeline real y exportar solo el panel, sin entrenar modelos."""
    from mibici.dataset import read_manifest, verify_manifest
    from mibici.features import full_weeks
    from mibici.geography import boundary_geometry, municipality_station_ids
    from mibici.modeling.p2 import weekly_counts
    from mibici.p2_panel import build_panel

    manifest = read_manifest(manifest_path)
    verify_manifest(manifest, data_dir)
    # La validación no descarga ni cambia fuentes.
    boundary_path = Path(data_dir) / "external" / "zapopan-inegi.geojson"
    geography = json.loads(config.P1_BOUNDARY_MANIFEST.read_text(encoding="utf-8"))
    if not boundary_path.is_file() or sha256(boundary_path) != geography["sha256"]:
        raise ValueError("Falta el polígono verificado. Prepáralo con python -m mibici.geography.")
    geometry = boundary_geometry(json.loads(boundary_path.read_text(encoding="utf-8")))
    stations = pd.read_parquet(Path(data_dir) / "processed" / "stations.parquet")
    ids = municipality_station_ids(stations, geometry)
    counts = weekly_counts(manifest, data_dir)
    panel, features, metadata = build_panel(counts, stations, full_weeks(manifest["months"]),
        config.P2_VALIDATION_WEEKS, config.P2_TEST_WEEKS, station_ids=ids)
    panel = panel[REQUIRED].copy()
    for name in ["week_start", "target_week"]:
        panel[name] = panel[name].dt.strftime("%Y-%m-%d")
    errors = panel_errors(panel)
    if errors:
        raise ValueError("Panel incompatible: " + " ".join(errors))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False, lineterminator="\n")
    provenance = {"kind": "real_p2_zapopan", "panel_sha256": sha256(output),
                  "data_version": manifest["data_version"], "data_manifest": manifest,
                  "geography_manifest": geography, "features": features,
                  "split_metadata": metadata}
    write_json(output.with_suffix(".metadata.json"), provenance)
    print(f"Exportado: {len(panel)} filas; {panel.station_id.nunique()} estaciones. {output}")


def demo_panel():
    """Datos sintéticos explícitos para comprobar la instalación, no evidencia MiBici."""
    rows = []
    for w, week in enumerate(pd.date_range("2025-01-06", periods=12, freq="7D")):
        for station in range(1, 13):
            departures, arrivals = 8 + station + w % 3, 6 + station
            rows.append({"station_id": station, "week_start": week.strftime("%Y-%m-%d"),
                         "target_week": (week + pd.Timedelta(weeks=1)).strftime("%Y-%m-%d"),
                         "split": "train" if w < 4 else "validation" if w < 8 else "test",
                         "target": (station + w) % 2, "capacity": 10 + station,
                         "departures": departures, "arrivals": arrivals,
                         "net_flow": arrivals - departures,
                         "departures_mean4": float(departures),
                         "arrivals_mean4": float(arrivals),
                         "net_flow_mean4": float(arrivals - departures)})
    return pd.DataFrame(rows, columns=REQUIRED)


def run_checks(panel, report_dir, provenance, strict_drift=False):
    """Suite estricta de integridad y diagnóstico separado de drift temporal."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {"source": provenance, "rows": len(panel), "schema_errors": panel_errors(panel),
               "integrity": {}, "drift": {}, "strict_drift": strict_drift}
    if summary["schema_errors"]:
        summary["status"] = "SCHEMA_FAILED"
        write_json(report_dir / "summary.json", summary)
        return 1, summary
    # Evitar la consulta de versión por red al importar Deepchecks.
    os.environ["DISABLE_LATEST_VERSION_CHECK"] = "True"
    from deepchecks.tabular import Dataset, Suite
    from deepchecks.tabular.checks.data_integrity import DataDuplicates, PercentOfNulls
    from deepchecks.tabular.checks.train_test_validation import FeatureDrift, LabelDrift

    summary["versions"] = {name: version(name) for name in
                            ["deepchecks", "numpy", "pandas", "scikit-learn"]}
    datasets = {}
    execution_failed = False
    for split in ["train", "validation", "test"]:
        frame = panel[panel["split"] == split].reset_index(drop=True)
        # Duplicados: misma entidad y semana, no meras features repetidas.
        data = Dataset(frame[[*KEYS, *FEATURES, "target"]], label="target",
                       cat_features=["station_id", "target_week"])
        suite = Suite(f"P2 integridad {split}",
            PercentOfNulls(columns=[*KEYS, *FEATURES, "target"],
                           n_samples=len(frame), random_state=config.RANDOM_SEED)
                .add_condition_percent_of_nulls_not_greater_than(0),
            DataDuplicates(columns=KEYS, n_samples=len(frame), random_state=config.RANDOM_SEED)
                .add_condition_ratio_less_or_equal(0))
        result = suite.run(data)
        result.save_as_html(str(report_dir / f"integrity_{split}.html"),
                            as_widget=False, connected=False)
        payload = result.to_json(with_display=False)
        (report_dir / f"integrity_{split}.json").write_text(
            payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
        not_run = len(result.get_not_ran_checks())
        execution_failed |= bool(not_run)
        summary["integrity"][split] = {"passed": bool(result.passed(
            fail_if_warning=True, fail_if_check_not_run=True)), "checks_not_run": not_run,
            "rows": len(frame), "positive_rows": int(frame.target.sum()) if
            frame.target.notna().all() else None}
        # Nunca usar IDs o fechas como variables para medir drift.
        datasets[split] = Dataset(frame[[*FEATURES, "target"]], label="target", cat_features=[])
    for split in ["validation", "test"]:
        drift = Suite(f"P2 drift train vs {split}",
            FeatureDrift(n_samples=len(panel), random_state=config.RANDOM_SEED,
                         numerical_drift_method="KS")
                .add_condition_drift_score_less_than(
                    max_allowed_numeric_score=config.DEEPCHECKS_DRIFT_THRESHOLD,
                    max_allowed_categorical_score=config.DEEPCHECKS_DRIFT_THRESHOLD),
            LabelDrift(n_samples=len(panel), random_state=config.RANDOM_SEED)
                .add_condition_drift_score_less_than(config.DEEPCHECKS_DRIFT_THRESHOLD))
        result = drift.run(datasets["train"], datasets[split])
        result.save_as_html(str(report_dir / f"drift_train_{split}.html"),
                            as_widget=False, connected=False)
        payload = result.to_json(with_display=False)
        (report_dir / f"drift_train_{split}.json").write_text(
            payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
        not_run = len(result.get_not_ran_checks())
        execution_failed |= bool(not_run)
        summary["drift"][split] = {"within_threshold": bool(result.passed(
            fail_if_warning=True, fail_if_check_not_run=True)), "checks_not_run": not_run}
    integrity_ok = all(r["passed"] for r in summary["integrity"].values())
    drift_ok = all(r["within_threshold"] for r in summary["drift"].values())
    code = 3 if execution_failed else 1 if not integrity_ok else 2 if strict_drift and not drift_ok else 0
    summary["status"] = ("CHECK_ERROR" if execution_failed else "INTEGRITY_FAILED" if
                         not integrity_ok else "PASS" if drift_ok else "DRIFT_REQUIRES_REVIEW")
    write_json(report_dir / "summary.json", summary)
    return code, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-p2", help="Exportar el panel real con el entorno ML")
    export.add_argument("--manifest", type=Path, default=config.P2_DATA_MANIFEST)
    export.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    export.add_argument("--output", type=Path, default=config.DEEPCHECKS_PANEL_PATH)
    check = commands.add_parser("check", help="Validar el CSV sin cargar modelos")
    check.add_argument("--panel", type=Path, default=config.DEEPCHECKS_PANEL_PATH)
    check.add_argument("--demo", action="store_true", help="Usar datos sintéticos, no MiBici")
    check.add_argument("--strict-drift", action="store_true")
    check.add_argument("--report-dir", type=Path, default=config.DEEPCHECKS_REPORT_DIR)
    args = parser.parse_args(argv)
    if args.command == "export-p2":
        export_p2(args.manifest, args.data_dir, args.output)
        return 0
    folder = args.report_dir / (("demo_" if args.demo else "real_") +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    try:
        if args.demo:
            panel, source = demo_panel(), {"kind": "synthetic_demo_not_mibici"}
        else:
            source = json.loads(args.panel.with_suffix(".metadata.json").read_text(encoding="utf-8"))
            if sha256(args.panel) != source["panel_sha256"]:
                raise ValueError("El CSV cambió respecto de su metadata; vuelve a exportarlo.")
            panel = pd.read_csv(args.panel)
        code, summary = run_checks(panel, folder, source, args.strict_drift)
    except Exception as error:
        # Un fallo de librería también produce evidencia de ERROR, nunca PASS.
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "summary.json", {"status": "ERROR", "error": str(error)})
        print(f"No se completó Deepchecks: {error}\nRevisa {folder}", file=sys.stderr)
        return 3
    print(f"Deepchecks: {summary['status']}\nReportes: {folder}")
    if summary["schema_errors"]:
        print("\n".join(summary["schema_errors"]))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
