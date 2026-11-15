"""Referencia congelada P1 y lote posterior reservado para monitoreo y evaluación A/B."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mibici import config
from mibici.dataset import DataError, read_manifest, verify_manifest
from mibici.features import full_weeks, weekly_departures

FEATURES = config.P1_CANDIDATE_FEATURES
COLUMNS = ["station_id", "week_start", "departures", *FEATURES]
DEFAULT_PROTOCOL = config.PROJ_ROOT / "references/monitoring-p1-protocol.json"
DEFAULT_DATA = config.DATA_DIR / "monitoring/p1"


def dump_json(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def frame_digest(frame):
    """Hash lógico portable: orden fijo, nulos explícitos y floats redondeados a 9 decimales.

    Tolerancia de serialización, no redondeo de las variables utilizadas por el modelo.
    Evita depender de metadatos Parquet o del último bit trigonométrico de cada SO.
    """
    rows = []
    for row in frame[COLUMNS].itertuples(index=False, name=None):
        rows.append([int(row[0]), pd.Timestamp(row[1]).isoformat(), int(row[2]),
                     *[None if pd.isna(x) else round(float(x), 9) for x in row[3:]]])
    return hashlib.sha256(json.dumps(rows, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_frame(frame):
    if list(frame.columns) != COLUMNS or frame.empty:
        raise DataError("Esquema incorrecto o partición vacía de monitoreo.")
    if frame.duplicated(["station_id", "week_start"]).any():
        raise DataError("Claves estación/semana duplicadas.")
    if not pd.api.types.is_datetime64_any_dtype(frame.week_start):
        raise DataError("week_start debe ser fecha.")
    if frame.week_start.dt.dayofweek.ne(0).any() or not frame.week_start.eq(frame.week_start.dt.normalize()).all():
        raise DataError("Las semanas deben comenzar el lunes.")
    if frame[["station_id", "departures"]].isna().any().any():
        raise DataError("Claves y objetivo no admiten nulos.")
    if not np.isfinite(frame[["station_id", "departures"]].to_numpy(dtype=float)).all():
        raise DataError("Claves y objetivo deben ser finitos.")
    if not pd.api.types.is_integer_dtype(frame.station_id):
        raise DataError("station_id debe ser entero.")
    if ((frame.departures < 0) | (frame.departures % 1 != 0)).any():
        raise DataError("El objetivo debe ser conteo no negativo.")
    values = frame[FEATURES].to_numpy(dtype=float)
    if np.isinf(values).any() or frame[[f for f in FEATURES if f != "lag_52"]].isna().any().any():
        raise DataError("Sólo lag_52 admite historia ausente; infinitos prohibidos.")
    if (frame[["lag_1", "mean_previous_4", "lag_52"]] < 0).any().any():
        raise DataError("Retardos negativos.")
    if not np.allclose(frame.week_sin ** 2 + frame.week_cos ** 2, 1, atol=1e-12):
        raise DataError("Calendario cíclico incoherente.")


def current_panel(historical_counts, current_counts, original_panel, historical_months, current_months):
    """Extender sólo el historial de la cohorte original, sin recalcular cortes ni selección."""
    from mibici.modeling.p1_candidates import candidate_features

    months = [*historical_months, *current_months]
    weeks = full_weeks(months)  # comprueba continuidad entre ambos periodos
    target_weeks = full_weeks(current_months)
    if not len(target_weeks) or target_weeks[0] <= original_panel.week_start.max():
        raise DataError("Current debe contener semanas completas posteriores al histórico.")
    counts = pd.concat([historical_counts, current_counts]).groupby(
        ["station_id", "week_start"]).departures.sum()
    cohort = sorted(original_panel.station_id.unique())
    history_weeks = full_weeks(historical_months)
    frames = []
    for station in cohort:
        history = counts.loc[station]
        seen = history[(history.index.isin(history_weeks)) & (history > 0)]
        if seen.empty:
            raise DataError("Estación de la cohorte sin historia original.")
        station_weeks = weeks[weeks >= seen.index.min()]
        frame = pd.DataFrame({"week_start": station_weeks,
                              "departures": history.reindex(station_weeks, fill_value=0).to_numpy()})
        frame["station_id"] = int(station)
        frame["lag_1"] = frame.departures.shift(1)
        frame["mean_previous_4"] = frame.departures.shift(1).rolling(4, min_periods=4).mean()
        frame["lag_52"] = frame.departures.shift(52)
        frame = frame[frame.week_start.isin(target_weeks)].copy()
        frame[FEATURES] = candidate_features(frame)
        frames.append(frame[COLUMNS])
    result = pd.concat(frames).sort_values(["week_start", "station_id"]).reset_index(drop=True)
    validate_frame(result)
    return result


def prepare(protocol_path=DEFAULT_PROTOCOL, historical_dir=config.DATA_DIR,
            current_dir=config.DATA_DIR / "monitoring/source", destination=DEFAULT_DATA):
    from mibici.modeling.p1_candidates import prepare_panel, candidate_features

    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    old_path = config.PROJ_ROOT / protocol["historical_manifest"]
    new_path = config.PROJ_ROOT / protocol["current_manifest"]
    panel, old, split, geography = prepare_panel(old_path, historical_dir)
    new = read_manifest(new_path)
    verify_manifest(new, current_dir)
    if old["data_version"] != protocol["model_a"]["data_version"]:
        raise DataError("El histórico no corresponde al modelo A.")
    if new["data_version"] != protocol["current_data_version"]:
        raise DataError("Current no corresponde a la versión fijada.")
    for key, expected in protocol["frozen_split"].items():
        if split[key] != expected:
            raise DataError(f"Corte histórico distinto del entrenamiento A: {key}.")
    parts = {}
    for name, split_name in (("reference", "train"), ("historical_validation", "validation"),
                             ("historical_test", "test")):
        frame = panel.loc[panel.split.eq(split_name)].copy()
        frame[FEATURES] = candidate_features(frame)
        parts[name] = frame[COLUMNS].reset_index(drop=True)
    parts["current"] = current_panel(
        weekly_departures(old, historical_dir), weekly_departures(new, current_dir),
        panel, old["months"], new["months"])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {"schema_version": 1, "model_a": protocol["model_a"],
                "features": FEATURES, "target": "departures", "keys": ["station_id", "week_start"],
                "historical_data_version": old["data_version"], "current_data_version": new["data_version"],
                "historical_manifest": protocol["historical_manifest"],
                "current_manifest": protocol["current_manifest"], "historical_split": split,
                "cohort_station_ids": sorted(map(int, panel.station_id.unique())),
                "evaluation_policy": "all_current_reserved_no_training_or_tuning",
                "metrics": {"primary": "mae", "secondary": ["rmse", "wape"]},
                "digest_policy": "ordered_rows_dates_iso_float_round_9_nan_null",
                "source_quality": new["summary"], "partitions": {}}
    for name, frame in parts.items():
        validate_frame(frame)
        digest = frame_digest(frame)
        expected = protocol.get("partition_digests", {}).get(name)
        if expected and digest != expected:
            raise DataError(f"Contenido lógico distinto del contrato: {name}.")
        frame.to_parquet(destination / f"{name}.parquet", index=False)
        metadata["partitions"][name] = {
            "path": f"{name}.parquet", "rows": len(frame), "stations": int(frame.station_id.nunique()),
            "weeks": int(frame.week_start.nunique()), "start": str(frame.week_start.min().date()),
            "end_exclusive": str((frame.week_start.max() + pd.Timedelta(days=7)).date()),
            "logical_sha256": digest}
    metadata["contract_id"] = hashlib.sha256(json.dumps(
        metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    dump_json(metadata, destination / "metadata.json")
    return metadata


def load_partitions(data_dir=DEFAULT_DATA):
    root = Path(data_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    expected_id = metadata.pop("contract_id")
    actual_id = hashlib.sha256(json.dumps(
        metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    metadata["contract_id"] = expected_id
    if actual_id != expected_id:
        raise DataError("Metadatos de contrato modificados.")
    parts = {}
    if set(metadata["partitions"]) != {"reference", "historical_validation", "historical_test", "current"}:
        raise DataError("Faltan particiones obligatorias del contrato.")
    for name, record in metadata["partitions"].items():
        path = root / record["path"]
        if path.resolve().parent != root.resolve():
            raise DataError("Ruta fuera del contrato.")
        frame = pd.read_parquet(path)
        validate_frame(frame)
        if frame_digest(frame) != record["logical_sha256"] or len(frame) != record["rows"]:
            raise DataError(f"Partición ausente o modificada: {name}.")
        parts[name] = frame
    if parts["reference"].week_start.max() >= parts["current"].week_start.min():
        raise DataError("Solapamiento temporal de referencia/current.")
    return parts, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify", "report"])
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--historical-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--current-dir", type=Path, default=config.DATA_DIR / "monitoring/source")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--report-dir", type=Path, default=config.REPORTS_DIR / "monitoring/p1")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.protocol, args.historical_dir, args.current_dir, args.data_dir)
        elif args.command == "verify":
            _, result = load_partitions(args.data_dir)
        else:
            from mibici.monitoring_report import create_report
            result = create_report(args.data_dir, args.report_dir)
    except (DataError, OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
