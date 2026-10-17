"""Agregaciones y variables reutilizables de P1, disponibles al origen de cada semana."""

import pandas as pd
import pyarrow.parquet as pq

from mibici import config
from mibici.dataset import DataError, manifest_path


def weekly_departures(manifest, data_dir):
    """Contar inicios por estación-semana leyendo únicamente dos columnas por lotes.

    La semana comienza el lunes a las 00:00 en la hora publicada por MiBici;
    no se infiere zona horaria. Esta agregación define el objetivo observado de P1.
    """
    partial = []
    for entry in manifest["files"]:
        if entry["kind"] != "trips":
            continue
        parquet = pq.ParquetFile(manifest_path(data_dir, entry["path"]))
        for batch in parquet.iter_batches(batch_size=config.CHUNK_SIZE,
                                          columns=["origin_station_id", "started_at"]):
            frame = batch.to_pandas().rename(columns={"origin_station_id": "station_id"})
            frame["week_start"] = (frame.started_at.dt.normalize()
                                    - pd.to_timedelta(frame.started_at.dt.dayofweek, unit="D"))
            counts = frame.groupby(["station_id", "week_start"]).size().rename("departures")
            partial.append(counts)
    if not partial:
        raise DataError("No hay archivos de viajes para construir P1.")
    result = pd.concat(partial).groupby(level=[0, 1]).sum().reset_index()
    return result.sort_values(["station_id", "week_start"]).reset_index(drop=True)


def full_weeks(months):
    """Excluir semanas cortadas por el principio o fin del periodo descargado."""
    periods = pd.PeriodIndex(months, freq="M")
    expected = pd.period_range(periods.min(), periods.max(), freq="M")
    if list(periods) != list(expected):
        raise DataError("P1 requiere meses consecutivos ordenados sin huecos.")
    start = periods.min().start_time.normalize()
    end = (periods.max() + 1).start_time.normalize()
    first = start + pd.Timedelta(days=(-start.dayofweek) % 7)
    stop = end - pd.Timedelta(days=end.dayofweek)
    return pd.date_range(first, stop - pd.Timedelta(days=7), freq="7D")


def p1_station_week_panel(counts, station_ids, months,
                          validation_weeks=config.P1_VALIDATION_WEEKS,
                          test_weeks=config.P1_TEST_WEEKS,
                          history_weeks=config.P1_HISTORY_WEEKS, cohort_start_month=None):
    """Construir cohorte histórica y retardos sin usar observaciones de la semana objetivo.

    La cohorte se fija con estaciones que ya tienen salidas antes de validación.
    Se completan ceros desde su primera semana observada en entrenamiento; no se
    inventa actividad antes de esa observación. Cero significa ningún inicio
    observado, no estación abierta, inventario disponible ni demanda potencial.
    """
    if min(validation_weeks, test_weeks, history_weeks) < 1:
        raise DataError("Las ventanas de P1 deben ser positivas.")
    weeks = full_weeks(months)
    if len(weeks) <= validation_weeks + test_weeks + history_weeks:
        raise DataError("No hay suficientes semanas completas para entrenamiento/validación/prueba.")
    validation_start = weeks[-(validation_weeks + test_weeks)]
    test_start = weeks[-test_weeks]
    selected = counts[counts.station_id.isin(station_ids) & counts.week_start.isin(weeks)].copy()
    past = selected[(selected.week_start < validation_start) & (selected.departures > 0)]
    if cohort_start_month is not None:
        cohort_start = pd.Period(cohort_start_month, freq="M").start_time
        cohort_start += pd.Timedelta(days=(-cohort_start.dayofweek) % 7)
        recent = past[past.week_start >= cohort_start]
    else:
        cohort_start = weeks[0]
        recent = past
    recent_first_seen = recent.groupby("station_id").week_start.min()
    first_seen = past[past.station_id.isin(recent_first_seen.index)].groupby("station_id").week_start.min()
    if first_seen.empty:
        raise DataError("No hay estaciones de Zapopan con historia de entrenamiento.")
    pieces = []
    for station_id, first_week in first_seen.items():
        history = selected[selected.station_id == station_id].set_index("week_start").departures
        station_weeks = weeks[weeks >= first_week]
        frame = pd.DataFrame({"week_start": station_weeks,
                              "departures": history.reindex(station_weeks, fill_value=0).to_numpy()})
        frame["station_id"] = int(station_id)
        frame["lag_1"] = frame.departures.shift(1)
        frame["lag_52"] = frame.departures.shift(config.P1_SEASONAL_WEEKS)
        frame["evaluation_eligible_from"] = (recent_first_seen.loc[station_id]
                                             + pd.Timedelta(weeks=history_weeks))
        frame[f"mean_previous_{history_weeks}"] = (frame.departures.shift(1)
                                                   .rolling(history_weeks, min_periods=history_weeks).mean())
        pieces.append(frame)
    panel = pd.concat(pieces, ignore_index=True)
    history_column = f"mean_previous_{history_weeks}"
    panel = panel.dropna(subset=["lag_1", history_column]).copy()
    panel["split"] = "train"
    panel.loc[panel.week_start >= validation_start, "split"] = "validation"
    panel.loc[panel.week_start >= test_start, "split"] = "test"
    panel = panel[(panel.split == "train") | (panel.week_start >= panel.evaluation_eligible_from)].copy()
    panel = panel.drop(columns="evaluation_eligible_from")
    if any(not (panel.split == split).any() for split in ("train", "validation", "test")):
        raise DataError("Una partición no tiene ejemplos tras requerir historia suficiente.")
    panel = panel.sort_values(["week_start", "station_id"]).reset_index(drop=True)
    metadata = {"full_weeks": len(weeks), "first_full_week": weeks[0].date().isoformat(),
                "last_full_week": weeks[-1].date().isoformat(),
                "validation_start": validation_start.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end_exclusive": (weeks[-1] + pd.Timedelta(days=7)).date().isoformat(),
                "history_weeks": history_weeks, "cohort_stations": len(first_seen),
                "geographic_catalog_stations": len(station_ids),
                "new_station_ids_excluded": sorted(set(selected.station_id) - set(first_seen.index)),
                "rows_by_split": panel.groupby("split").size().astype(int).to_dict(),
                "stations_by_split": panel.groupby("split").station_id.nunique().astype(int).to_dict(),
                "weeks_by_split": panel.groupby("split").week_start.nunique().astype(int).to_dict(),
                "target": "observed_started_trips", "horizon_days": 7,
                "evaluation_protocol": "weekly_rolling_origin_prior_observations_available",
                "cohort_policy": "observed_departures_in_recent_training_only",
                "cohort_observation_start": cohort_start.date().isoformat()}
    return panel, metadata
