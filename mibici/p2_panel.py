"""Panel retrospectivo P2. Solo pandas; independiente de ingesta y tracking."""

import pandas as pd


FEATURES = ["capacity", "departures", "arrivals", "net_flow",
            "departures_mean4", "arrivals_mean4", "net_flow_mean4"]


def build_panel(counts, stations, weeks, validation_weeks=8, test_weeks=8, station_ids=None):
    """Fijar cohorte con historia de entrenamiento y predecir t+1 desde cierre de t.

    Los ceros posteriores a la primera actividad son ausencia de viajes registrados,
    no evidencia de operación. El catálogo es una referencia retrospectiva fija.
    """
    weeks = pd.DatetimeIndex(weeks)
    if validation_weeks < 1 or test_weeks < 1:
        raise ValueError("Validación y prueba deben tener al menos una semana.")
    if len(weeks) <= validation_weeks + test_weeks + 5:
        raise ValueError("Faltan semanas completas para historia y tres particiones.")
    if not weeks.equals(pd.date_range(weeks.min(), weeks.max(), freq="7D")):
        raise ValueError("Se requieren semanas consecutivas y ordenadas.")
    if (weeks.dayofweek != 0).any():
        raise ValueError("Las semanas deben comenzar en lunes.")
    if stations.station_id.duplicated().any():
        raise ValueError("El catálogo debe tener station_id único.")
    eligible = stations.loc[stations.capacity.notna() & (stations.capacity > 0),
                            ["station_id", "capacity"]].copy()
    geographic_ids = (set(stations.station_id) if station_ids is None else set(station_ids))
    eligible = eligible[eligible.station_id.isin(geographic_ids)]
    validation_start = weeks[-(validation_weeks + test_weeks)]
    test_start = weeks[-test_weeks]
    selected = counts[counts.week_start.isin(weeks) &
                      counts.station_id.isin(eligible.station_id)].copy()
    if selected.duplicated(["station_id", "week_start"]).any():
        raise ValueError("Los conteos deben ser únicos por estación-semana.")
    # Una estación debe disponer de cuatro semanas completas antes de validación.
    historical = selected[(selected.week_start < validation_start) &
                          ((selected.departures + selected.arrivals) > 0)]
    first_seen = historical.groupby("station_id").week_start.min()
    first_seen = first_seen[first_seen + pd.Timedelta(weeks=4) <= validation_start]
    pieces = []
    for station_id, first_week in first_seen.items():
        frame = selected[selected.station_id == station_id].set_index("week_start")
        frame = frame[["departures", "arrivals"]].reindex(weeks[weeks >= first_week],
                                                          fill_value=0).fillna(0)
        frame.index.name = "week_start"
        frame["station_id"] = station_id
        frame["capacity"] = eligible.set_index("station_id").loc[station_id, "capacity"]
        frame["net_flow"] = frame.arrivals - frame.departures
        frame["risk"] = ((frame.departures - frame.arrivals) >= frame.capacity).astype(int)
        for column in ("departures", "arrivals", "net_flow"):
            frame[f"{column}_mean4"] = frame[column].rolling(4, min_periods=4).mean()
        frame["target"] = frame.risk.shift(-1)
        frame["target_week"] = frame.index + pd.Timedelta(weeks=1)
        pieces.append(frame.reset_index().dropna(subset=FEATURES + ["target"]))
    if not pieces:
        raise ValueError("Ninguna estación tiene historia suficiente de entrenamiento.")
    panel = pd.concat(pieces, ignore_index=True)
    panel["target"] = panel.target.astype(int)
    panel["split"] = "train"
    panel.loc[panel.target_week >= validation_start, "split"] = "validation"
    panel.loc[panel.target_week >= test_start, "split"] = "test"
    panel = panel.sort_values(["target_week", "station_id"]).reset_index(drop=True)
    metadata = {
        "scope": ("all_network_catalog_positive_capacity_historical_cohort" if station_ids is None
                  else "reference_polygon_positive_capacity_historical_cohort"),
        "capacity_policy": "fixed_september_2026_reference_not_historical_inventory",
        "history_weeks": 4, "horizon_days": 7,
        "forecast_origin": "target_monday_00_source_naive_time",
        "evaluation_protocol": "fixed_train_fit_weekly_observed_features_no_test_tuning",
        "cohort_stations": len(first_seen),
        "catalog_stations": len(stations),
        "geographic_catalog_stations": int(stations.station_id.isin(geographic_ids).sum()),
        "catalog_nonpositive_or_missing_capacity": int(
            (stations.capacity.isna() | (stations.capacity <= 0)).sum()),
        "geographic_nonpositive_or_missing_capacity": int(
            (stations.station_id.isin(geographic_ids) &
             (stations.capacity.isna() | (stations.capacity <= 0))).sum()),
        "unknown_station_ids": sorted(int(i) for i in set(counts.station_id) - set(stations.station_id)),
        "stations_without_sufficient_training_history": sorted(
            int(i) for i in set(eligible.station_id) - set(first_seen.index)),
        "first_full_week": weeks[0].date().isoformat(),
        "last_full_week": weeks[-1].date().isoformat(),
        "validation_start": validation_start.date().isoformat(),
        "test_start": test_start.date().isoformat(),
        "test_end_exclusive": (weeks[-1] + pd.Timedelta(weeks=1)).date().isoformat(),
        "partitions": {},
    }
    for split in ("train", "validation", "test"):
        frame = panel[panel.split == split]
        if frame.empty:
            raise ValueError(f"Partición vacía: {split}.")
        metadata["partitions"][split] = {
            "rows": len(frame), "stations": int(frame.station_id.nunique()),
            "positive_rows": int(frame.target.sum()),
            "target_start": frame.target_week.min().date().isoformat(),
            "target_end_exclusive": (frame.target_week.max() + pd.Timedelta(weeks=1)).date().isoformat(),
        }
    return panel, list(FEATURES), metadata
