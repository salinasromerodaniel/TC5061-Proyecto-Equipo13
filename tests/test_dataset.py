"""Pruebas con datos sintéticos; ningún registro real se incluye en el repositorio."""

import shutil

import pandas as pd
import pytest

from mibici import dataset as ds


def trips():
    return pd.DataFrame({
        "Viaje_Id": [1, 2], "Usuario_Id": [111, 222], "Genero": ["X", "X"],
        "Año_de_nacimiento": [1990, 1991],
        "Inicio_del_viaje": ["2026-09-01 10:00:00", "2026-09-01 11:00:00"],
        "Fin_del_viaje": ["2026-09-01 10:10:00", "2026-09-01 11:15:00"],
        "Origen_Id": [2, 3], "Destino_Id": [3, 2],
    }).astype("string")


def stations():
    return pd.DataFrame({"id": [2, 3], "name": ["Estación A", "ex-Estación B"],
                         "latitude": [20.6, 20.7], "longitude": [-103.3, -103.4],
                         "dpcapacity": [15, 0]}).astype("string")


def test_month_range():
    assert ds.month_range("2025-12", "2026-02") == ["2025-12", "2026-01", "2026-02"]


@pytest.mark.parametrize("start,end", [("2026-13", "2026-13"), ("2026-9", "2026-10"),
                                       ("2026-10", "2026-09")])
def test_invalid_months(start, end):
    with pytest.raises(ds.DataError):
        ds.month_range(start, end)


def test_normalization_excludes_personal_fields_and_handles_headers():
    frame = trips().rename(columns={"Viaje_Id": " VIAJE ID ", "Origen_Id": "Origen ID"})
    clean = ds.normalize_trips(frame, "2026-09")
    assert clean.columns.tolist() == ds.TRIP_SCHEMA.names
    assert "Usuario_Id" not in clean and "Año_de_nacimiento" not in clean
    assert str(clean.started_at.dtype) == "datetime64[ns]"
    assert clean.trip_id.tolist() == [1, 2]


@pytest.mark.parametrize("column,value", [("Viaje_Id", "1.5"), ("Origen_Id", ""),
                                          ("Inicio_del_viaje", "2026-08-31 23:59:59"),
                                          ("Fin_del_viaje", "2026-09-01 09:00:00")])
def test_reject_invalid_trips(column, value):
    frame = trips()
    frame.loc[0, column] = value
    with pytest.raises(ds.DataError):
        ds.normalize_trips(frame, "2026-09")


def test_day_first_date_explicit():
    result = ds.datetime_column(pd.Series(["02/09/2026 10:00:00"]), "date")
    assert result.iloc[0] == pd.Timestamp("2026-09-02 10:00:00")


def test_ambiguous_columns_rejected():
    frame = trips()
    frame["trip_id"] = [1, 2]
    with pytest.raises(ds.DataError):
        ds.normalize_trips(frame, "2026-09")


def test_station_history_preserved():
    clean = ds.normalize_stations(stations())
    assert len(clean) == 2 and clean.capacity.tolist() == [15, 0]
    assert clean.station_name.iloc[1].startswith("ex-")


@pytest.mark.parametrize("column,value", [("latitude", "100"), ("dpcapacity", "-1")])
def test_invalid_stations(column, value):
    frame = stations()
    frame.loc[0, column] = value
    with pytest.raises(ds.DataError):
        ds.normalize_stations(frame)


def test_station_duplicate_rejected():
    frame = stations()
    frame.loc[1, "id"] = frame.loc[0, "id"]
    with pytest.raises(ds.DataError):
        ds.normalize_stations(frame)


def test_cp1252_and_semicolon(tmp_path):
    path = tmp_path / "stations.csv"
    stations().to_csv(path, index=False, encoding="cp1252", sep=";")
    encoding, delimiter = ds.csv_format(path)
    assert encoding == "cp1252" and delimiter == ";"
    clean = ds.normalize_stations(pd.read_csv(path, encoding=encoding, sep=delimiter))
    assert clean.station_name.iloc[0] == "Estación A"


def test_discover_real_directory_and_reject_missing():
    html = ('<a href="/site/assets/files/123/datos_abiertos_2026_09.csv">Sep</a>'
            '<a href="/site/assets/files/1118/nomenclatura_2026_09.csv">Estaciones</a>')
    urls = ds.source_urls(html, ["2026-09"], "nomenclatura_2026_09.csv")
    assert urls["datos_abiertos_2026_09.csv"].endswith("/123/datos_abiertos_2026_09.csv")
    with pytest.raises(ds.DataError):
        ds.source_urls(html, ["2026-08"], "nomenclatura_2026_09.csv")


def fixtures(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    trips().to_csv(source / "datos_abiertos_2026_09.csv", index=False)
    stations().to_csv(source / "nomenclatura_2026_09.csv", index=False)

    def copy_download(session, url, destination, expected_hash=None):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if expected_hash and destination.exists():
            if ds.sha256_file(destination) != expected_hash:
                raise ds.DataError("Hash de caché inválido")
            return
        shutil.copyfile(source / url.rsplit("/", 1)[-1], destination)
        if expected_hash and ds.sha256_file(destination) != expected_hash:
            raise ds.DataError("Hash de fuente inválido")

    monkeypatch.setattr(ds, "download", copy_download)
    urls = {"stations": "https://www.mibici.net/nomenclatura_2026_09.csv",
            "2026-09": "https://www.mibici.net/datos_abiertos_2026_09.csv"}
    return source, urls


def test_pipeline_deterministic_across_directories_and_detects_tampering(tmp_path, monkeypatch):
    _, sources = fixtures(tmp_path, monkeypatch)
    first = ds.build_manifest(None, sources, tmp_path / "one", ["2026-09"], 1)
    second = ds.build_manifest(None, sources, tmp_path / "two", ["2026-09"], 1, first)
    assert first == second
    assert first["summary"] == {"months": 1, "raw_trips": 2, "trips": 2,
                                "duplicate_rows_removed": 0, "catalog_stations": 2,
                                "trip_stations": 2, "station_ids_missing_from_catalog": []}
    ds.verify_manifest(first, tmp_path / "one")
    manifest_path = tmp_path / "manifest.json"
    ds.save_manifest(first, manifest_path)
    assert ds.read_manifest(manifest_path) == first
    broken = tmp_path / "one" / first["files"][0]["path"]
    broken.write_bytes(broken.read_bytes() + b"\n")
    with pytest.raises(ds.DataError):
        ds.verify_manifest(first, tmp_path / "one")
    with pytest.raises(ds.DataError):
        ds.build_manifest(None, sources, tmp_path / "one", ["2026-09"], 1, first)
    first["summary"]["trips"] = 999
    ds.save_manifest(first, manifest_path)
    with pytest.raises(ds.DataError):
        ds.read_manifest(manifest_path)


def test_conflicting_trip_across_chunks_rejected(tmp_path, monkeypatch):
    source, sources = fixtures(tmp_path, monkeypatch)
    frame = trips()
    frame.loc[1, "Viaje_Id"] = frame.loc[0, "Viaje_Id"]
    frame.to_csv(source / "datos_abiertos_2026_09.csv", index=False)
    with pytest.raises(ds.DataError, match="contradictorios"):
        ds.build_manifest(None, sources, tmp_path / "out", ["2026-09"], 1)
    assert not (tmp_path / "out/processed/trips_2026-09.parquet").exists()


def test_unknown_stations_reported_without_dropping_rows(tmp_path, monkeypatch):
    source, sources = fixtures(tmp_path, monkeypatch)
    frame = trips()
    frame.loc[1, "Destino_Id"] = "999"
    frame.to_csv(source / "datos_abiertos_2026_09.csv", index=False)
    result = ds.build_manifest(None, sources, tmp_path / "out", ["2026-09"], 1)
    assert result["summary"]["station_ids_missing_from_catalog"] == [999]
    assert result["summary"]["trips"] == 2


def test_path_escape_rejected(tmp_path):
    with pytest.raises(ds.DataError):
        ds.manifest_path(tmp_path, "../escape")


def test_exact_duplicates_removed_across_chunks_and_counted(tmp_path, monkeypatch):
    source, sources = fixtures(tmp_path, monkeypatch)
    frame = trips()
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(
        source / "datos_abiertos_2026_09.csv", index=False)
    result = ds.build_manifest(None, sources, tmp_path / "out", ["2026-09"], 1)
    assert result["summary"]["raw_trips"] == 3
    assert result["summary"]["trips"] == 2
    assert result["summary"]["duplicate_rows_removed"] == 1
    ds.verify_manifest(result, tmp_path / "out")


def test_integer_overflow_rejected():
    with pytest.raises(ds.DataError, match="int64"):
        ds.integer_column(pd.Series(["9223372036854775808"]), "id")


def test_download_hash_mismatch_preserves_previous_file(tmp_path):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, size): yield b"unexpected source"

    class Session:
        def get(self, *args, **kwargs): return Response()

    target = tmp_path / "download.csv"
    with pytest.raises(ds.DataError, match="fuente"):
        ds.download(Session(), "https://example.invalid/data.csv", target, "0" * 64)
    assert not target.exists()
    assert not list(tmp_path.glob("*.part"))
    target.write_bytes(b"altered cache")
    with pytest.raises(ds.DataError, match="local"):
        ds.download(Session(), "https://example.invalid/data.csv", target, "0" * 64)
    assert target.read_bytes() == b"altered cache"


def test_historical_day_first_minute_precision():
    result = ds.datetime_column(pd.Series(["01/10/2024 0:00", "31/10/2024 23:59"]), "date")
    assert result.tolist() == [pd.Timestamp("2024-10-01 00:00"), pd.Timestamp("2024-10-31 23:59")]
