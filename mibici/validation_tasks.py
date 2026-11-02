"""Contratos y suites de P1/P3; orquestación de calidad para las tres preguntas.

P3 valida la matriz descriptiva completa: no inventa etiquetas ni particiones.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from mibici import config
from mibici.validation import demo_panel, export_p2, run_checks, sha256, write_json

P1_FEATURES = config.P1_CANDIDATE_FEATURES
P3_FEATURES = ["departures", "arrivals", "traffic", "balance", "capacity"]


def contract(question):
    if question == "P1":
        return (["station_id", "week_start"], P1_FEATURES, "target", ["lag_52"])
    if question == "P3":
        return (["station_id"], P3_FEATURES, None, [])
    raise ValueError(f"Pregunta sin contrato: {question}")


def required_columns(question):
    keys, features, label, _ = contract(question)
    return [*keys, *features, *(["split", label] if label else [])]


def frame_errors(frame, question):
    """Contrato previo; conserva los nulos numéricos para medirlos en Deepchecks."""
    keys, features, label, _ = contract(question)
    expected = required_columns(question)
    if frame.columns.duplicated().any():
        return ["Nombres de columnas duplicados."]
    if set(frame.columns) != set(expected):
        return [f"Columnas incompatibles: faltan {sorted(set(expected)-set(frame.columns))}; "
                f"sobran {sorted(set(frame.columns)-set(expected))}."]
    if frame.empty:
        return ["Dataset vacío."]
    errors = []
    for name in ["station_id", *features, *([label] if label else [])]:
        values = frame[name]
        if not is_numeric_dtype(values) or is_bool_dtype(values):
            errors.append(f"{name} requiere tipo numérico.")
        elif np.isinf(values.dropna().to_numpy(dtype=float)).any():
            errors.append(f"{name} contiene infinitos.")
    if errors:
        return errors
    ids = frame.station_id.dropna()
    if ((ids <= 0) | (ids % 1 != 0)).any():
        errors.append("station_id requiere enteros positivos.")
    for name in [*features, *([label] if label else [])]:
        if name not in ["balance", "week_sin", "week_cos"] and (frame[name].dropna() < 0).any():
            errors.append(f"{name} no puede ser negativo.")
    if question == "P1":
        dates = pd.to_datetime(frame.week_start, format="%Y-%m-%d", errors="coerce")
        if dates.isna().any() or (dates.dt.dayofweek != 0).any():
            errors.append("week_start requiere lunes ISO válidos.")
        if not frame["split"].isin(["train", "validation", "test"]).all():
            errors.append("Particiones inválidas.")
        for split in ["train", "validation", "test"]:
            if frame["split"].eq(split).sum() < config.DEEPCHECKS_MIN_ROWS:
                errors.append(f"{split} no tiene suficientes filas.")
        if not errors:
            for a, b in [("train", "validation"), ("validation", "test")]:
                if dates[frame["split"].eq(a)].max() >= dates[frame["split"].eq(b)].min():
                    errors.append(f"Solapamiento temporal entre {a} y {b}.")
        for name in ["week_sin", "week_cos"]:
            if (frame[name].dropna().abs() > 1 + 1e-10).any():
                errors.append(f"{name} fuera de [-1, 1].")
        if (frame.target.dropna() % 1 != 0).any():
            errors.append("El objetivo P1 requiere conteos enteros.")
    else:
        if len(frame) < config.DEEPCHECKS_MIN_ROWS:
            errors.append("P3 no tiene suficientes estaciones.")
        for name, actual in [("traffic", frame.departures + frame.arrivals),
                             ("balance", frame.departures - frame.arrivals)]:
            mask = frame[name].notna() & actual.notna()
            if not np.allclose(frame.loc[mask, name], actual[mask]):
                errors.append(f"{name} no coincide con salidas/llegadas.")
    return errors


def export_question(question, manifest_path, data_dir, output):
    """Reutilizar funciones reales sin entrenar ni registrar runs de MLflow."""
    if question == "P2":
        export_p2(manifest_path, data_dir, output)
        return
    from mibici.dataset import read_manifest, verify_manifest
    from mibici.features import p1_station_week_panel, weekly_departures, p3_station_features
    manifest = read_manifest(manifest_path)
    verify_manifest(manifest, data_dir)
    metadata = {}
    if question == "P1":
        from mibici.geography import boundary_geometry, municipality_station_ids
        from mibici.modeling.p1_candidates import candidate_features
        geography = json.loads(config.P1_BOUNDARY_MANIFEST.read_text(encoding="utf-8"))
        boundary = Path(data_dir) / "external/zapopan-inegi.geojson"
        if sha256(boundary) != geography["sha256"]:
            raise ValueError("Polígono INEGI incompatible con el manifiesto.")
        geometry = boundary_geometry(json.loads(boundary.read_text(encoding="utf-8")))
        stations = pd.read_parquet(Path(data_dir)/"processed/stations.parquet")
        panel, split = p1_station_week_panel(weekly_departures(manifest, data_dir),
            municipality_station_ids(stations, geometry), manifest["months"],
            cohort_start_month=config.P1_COHORT_START_MONTH)
        frame = panel[["station_id", "week_start", "split"]].copy()
        frame[P1_FEATURES] = candidate_features(panel)
        frame["target"] = panel.departures
        frame["week_start"] = frame.week_start.dt.strftime("%Y-%m-%d")
        metadata = {"split_metadata": split, "geography_manifest": geography,
                    "missing_policy": "lag_52 may be missing before 52 weeks; HGB accepts NaN; "
                                      "seasonal baseline falls back to lag_1"}
    else:
        frame = p3_station_features(manifest, Path(data_dir))[required_columns("P3")].copy()
        metadata = {"scope": "full_station_catalog_full_period_as_in_run_p3",
                    "drift": "not_applicable_no_train_test_split",
                    "upstream_policy": "p3_station_features fills missing values with zero; "
                                       "post-transform checks cannot detect original missingness"}
    errors = frame_errors(frame, question)
    if errors:
        raise ValueError(" ".join(errors))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, lineterminator="\n")
    write_json(output.with_suffix(".metadata.json"), {
        "kind": f"real_{question.lower()}", "question": question,
        "panel_sha256": sha256(output), "data_version": manifest["data_version"],
        "data_manifest": manifest, **metadata})
    print(f"{question}: exportadas {len(frame)} filas a {output}")


def synthetic_frame(question):
    if question == "P2":
        return demo_panel()
    base = demo_panel()
    if question == "P3":
        frame = base.drop_duplicates("station_id")[["station_id", "departures", "arrivals", "capacity"]].copy()
        frame["traffic"] = frame.departures + frame.arrivals
        frame["balance"] = frame.departures - frame.arrivals
    else:
        frame = base[["station_id", "week_start", "split"]].copy()
        frame["lag_1"] = base.departures.astype(float)
        frame["mean_previous_4"] = base.departures.astype(float)
        frame["lag_52"] = np.nan  # Historia estacional ausente, esperada por el contrato.
        phase = 2 * np.pi * (pd.to_datetime(frame.week_start).dt.dayofyear - 1) / 365.2425
        frame["week_sin"], frame["week_cos"] = np.sin(phase), np.cos(phase)
        frame["target"] = base.departures
    return frame[required_columns(question)]


def run_question(frame, question, report_dir, source, strict_drift=False):
    if question == "P2":
        return run_checks(frame, report_dir, source, strict_drift)
    from importlib.metadata import version
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    errors = frame_errors(frame, question)
    keys, features, label, allowed_nulls = contract(question)
    summary = {"question": question, "source": source, "rows": len(frame),
        "schema_errors": errors, "integrity": {}, "drift": {},
        "missing_counts": {name: int(frame[name].isna().sum()) for name in frame.columns},
        "allowed_missing_features": allowed_nulls}
    if errors:
        summary["status"] = "SCHEMA_FAILED"
        write_json(report_dir/"summary.json", summary)
        return 1, summary
    os.environ["DISABLE_LATEST_VERSION_CHECK"] = "True"
    from deepchecks.tabular import Dataset, Suite
    from deepchecks.tabular.checks.data_integrity import DataDuplicates, PercentOfNulls
    from deepchecks.tabular.checks.train_test_validation import FeatureDrift, LabelDrift
    summary["versions"] = {n: version(n) for n in ["deepchecks", "numpy", "pandas", "scikit-learn"]}
    not_run = False

    def save(result, name):
        nonlocal not_run
        result.save_as_html(str(report_dir/f"{name}.html"), as_widget=False, connected=False)
        payload = result.to_json(with_display=False)
        (report_dir/f"{name}.json").write_text(payload if isinstance(payload, str) else
                                             json.dumps(payload), encoding="utf-8")
        count = len(result.get_not_ran_checks())
        not_run |= bool(count)
        return {"passed": bool(result.passed(fail_if_warning=True, fail_if_check_not_run=True)),
                "checks_not_run": count}

    datasets = {}
    splits = ["train", "validation", "test"] if label else ["full"]
    for split in splits:
        subset = frame[frame["split"].eq(split)].reset_index(drop=True) if label else frame
        columns = [*keys, *features, *([label] if label else [])]
        data = Dataset(subset[columns], label=label, cat_features=keys,
                       **({"label_type": "regression"} if label else {}))
        # lag_52 no se imputa: HGB acepta NaN, igual que el pipeline original.
        mandatory = [n for n in columns if n not in allowed_nulls]
        checks = [PercentOfNulls(columns=mandatory, n_samples=len(subset), random_state=config.RANDOM_SEED)
                    .add_condition_percent_of_nulls_not_greater_than(0),
                  DataDuplicates(columns=keys, n_samples=len(subset), random_state=config.RANDOM_SEED)
                    .add_condition_ratio_less_or_equal(0)]
        if allowed_nulls:
            checks.append(PercentOfNulls(columns=allowed_nulls, n_samples=len(subset),
                                        random_state=config.RANDOM_SEED))
        summary["integrity"][split] = save(Suite(f"{question} integridad {split}", *checks).run(data),
                                              f"integrity_{split}")
        datasets[split] = Dataset(subset[[*features, *([label] if label else [])]], label=label,
                                 cat_features=[], **({"label_type": "regression"} if label else {}))
    if label:
        for split in ["validation", "test"]:
            # La señal estacional inexistente en todas las filas de algún bloque
            # se registra, pero no tiene una distribución para comparar.
            usable = [n for n in features if frame.loc[frame["split"].eq("train"), n].notna().any()
                      and frame.loc[frame["split"].eq(split), n].notna().any()]
            summary.setdefault("drift_excluded_all_missing", {})[split] = sorted(set(features)-set(usable))
            suite = Suite(f"{question} drift train vs {split}",
                FeatureDrift(columns=usable, n_samples=len(frame), random_state=config.RANDOM_SEED,
                             numerical_drift_method="KS").add_condition_drift_score_less_than(
                                 max_allowed_numeric_score=config.DEEPCHECKS_DRIFT_THRESHOLD,
                                 max_allowed_categorical_score=config.DEEPCHECKS_DRIFT_THRESHOLD),
                LabelDrift(n_samples=len(frame), random_state=config.RANDOM_SEED)
                    .add_condition_drift_score_less_than(config.DEEPCHECKS_DRIFT_THRESHOLD))
            summary["drift"][split] = save(suite.run(datasets["train"], datasets[split]), f"drift_train_{split}")
    else:
        summary["drift_not_applicable"] = "P3 is descriptive clustering of a full-period station matrix; no train/test datasets."
    integrity = all(v["passed"] for v in summary["integrity"].values())
    drift = all(v["passed"] for v in summary["drift"].values())
    code = 3 if not_run else 1 if not integrity else 2 if strict_drift and not drift else 0
    summary["status"] = ("CHECK_ERROR" if not_run else "INTEGRITY_FAILED" if not integrity else
                         "PASS" if drift else "DRIFT_REQUIRES_REVIEW")
    write_json(report_dir/"summary.json", summary)
    return code, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export", "check"])
    parser.add_argument("--question", choices=["P1", "P2", "P3", "all"], default="all")
    parser.add_argument("--manifest", type=Path, default=config.PROJECT_DATA_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--panel-dir", type=Path, default=config.PROCESSED_DATA_DIR/"quality")
    parser.add_argument("--report-dir", type=Path, default=config.DEEPCHECKS_REPORT_DIR)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--strict-drift", action="store_true")
    args = parser.parse_args(argv)
    questions = ["P1", "P2", "P3"] if args.question == "all" else [args.question]
    if args.demo and args.command == "export":
        parser.error("--demo se usa con check; export siempre utiliza datos reales.")
    folder = args.report_dir / (("export_" if args.command == "export" else "demo_" if args.demo else "real_") +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    results, codes = {}, []
    for question in questions:
        path = args.panel_dir/f"{question.lower()}_panel.csv"
        try:
            if args.command == "export":
                export_question(question, args.manifest, args.data_dir, path)
                results[question] = {"status": "EXPORTED", "panel_sha256": sha256(path)}
                codes.append(0)
                continue
            if args.demo:
                frame, source = synthetic_frame(question), {"kind": "synthetic_demo_not_mibici"}
            else:
                source = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
                if sha256(path) != source["panel_sha256"]:
                    raise ValueError("El panel cambió; volver a exportar los datos.")
                if source.get("kind") != {"P1":"real_p1", "P2":"real_p2_zapopan", "P3":"real_p3"}[question]:
                    raise ValueError("La procedencia no corresponde a la pregunta solicitada.")
                frame = pd.read_csv(path)
            code, result = run_question(frame, question, folder/question, source, args.strict_drift)
        except Exception as error:
            code, result = 3, {"status": "ERROR", "error": str(error)}
            print(f"{question}: {error}")
            if args.command == "check":
                (folder/question).mkdir(parents=True, exist_ok=True)
                write_json(folder/question/"summary.json", result)
        results[question] = result
        codes.append(code)
        print(f"{question}: {result['status']}")
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder/"summary.json", {"command": args.command,
        "kind": "demo" if args.demo else "real", "questions": results})
    print(f"Reportes: {folder}")
    return 3 if 3 in codes else 1 if 1 in codes else 2 if 2 in codes else 0


if __name__ == "__main__":
    raise SystemExit(main())
