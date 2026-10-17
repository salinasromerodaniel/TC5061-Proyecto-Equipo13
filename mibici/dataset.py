"""Descarga, normaliza y verifica datos de MiBici sin versionar registros en Git."""

import argparse
import codecs
import csv
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tempfile
import unicodedata
from urllib.parse import urljoin, urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from mibici import config

TRIP_SCHEMA = pa.schema([
    ("trip_id", pa.int64()),
    ("started_at", pa.timestamp("us")),
    ("ended_at", pa.timestamp("us")),
    ("origin_station_id", pa.int64()),
    ("destination_station_id", pa.int64()),
])
STATION_SCHEMA = pa.schema([
    ("station_id", pa.int64()), ("station_name", pa.string()),
    ("latitude", pa.float64()), ("longitude", pa.float64()), ("capacity", pa.int64()),
])
ALIASES = {
    "trip_id": ("viaje_id", "id_viaje", "trip_id"),
    "started_at": ("inicio_del_viaje", "fecha_inicio", "inicio", "started_at"),
    "ended_at": ("fin_del_viaje", "fecha_fin", "fin", "ended_at"),
    "origin_station_id": ("origen_id", "id_origen", "origin_station_id"),
    "destination_station_id": ("destino_id", "id_destino", "destination_station_id"),
    "station_id": ("id", "id_estacion", "station_id"),
    "station_name": ("name", "nombre", "station_name"),
    "latitude": ("latitude", "latitud"),
    "longitude": ("longitude", "longitud"),
    "capacity": ("dpcapacity", "capacidad", "capacity"),
}


class DataError(ValueError):
    """El archivo no permite reconstruir datos válidos y verificables."""


def sha256_file(path):
    """Calcular SHA-256 sin cargar archivos completos en memoria."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def data_version(manifest):
    """Versionar parámetros, fuentes y resultados, sin fechas de ejecución ni rutas absolutas."""
    return hashlib.sha256(canonical_bytes({k: v for k, v in manifest.items()
                                         if k != "data_version"})).hexdigest()


def month_range(start, end):
    """Producir meses inclusivos y rechazar rangos inválidos."""
    if not all(re.fullmatch(r"\d{4}-\d{2}", m) for m in (start, end)):
        raise DataError("Los meses deben tener formato YYYY-MM.")
    try:
        first, last = (datetime.strptime(m, "%Y-%m") for m in (start, end))
    except ValueError as exc:
        raise DataError("Mes inválido.") from exc
    if first > last:
        raise DataError("El mes inicial no puede ser posterior al final.")
    return pd.period_range(first, last, freq="M").astype(str).tolist()


def normalized_name(name):
    text = unicodedata.normalize("NFKD", str(name).lstrip("\ufeff").strip())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def column_mapping(columns, schema):
    """Resolver encabezados equivalentes sin asumir silenciosamente un esquema desconocido."""
    normalized = [normalized_name(c) for c in columns]
    if len(set(normalized)) != len(normalized):
        raise DataError("Hay columnas ambiguas tras normalizar encabezados.")
    available = dict(zip(normalized, columns))
    mapping = {}
    for field in schema.names:
        matches = [available[a] for a in ALIASES[field] if a in available]
        if len(matches) != 1:
            raise DataError(f"Se requiere una sola columna para {field}; encontradas: {matches}.")
        mapping[matches[0]] = field
    return mapping


def csv_format(path):
    """Validar codificación completa y detectar separador, sin sustituir caracteres inválidos."""
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        decoder = codecs.getincrementaldecoder(encoding)()
        try:
            with Path(path).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    decoder.decode(block)
                decoder.decode(b"", final=True)
            break
        except UnicodeDecodeError:
            continue
    with Path(path).open(encoding=encoding, newline="") as stream:
        sample = stream.read(8192)
    if not sample.strip() or sample.lstrip().lower().startswith(("<!doctype", "<html")):
        raise DataError("Se esperaba CSV y se recibió una respuesta vacía o HTML.")
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error as exc:
        raise DataError("No se pudo identificar el separador CSV.") from exc
    return encoding, delimiter


def integer_column(series, name, minimum=0):
    values = pd.to_numeric(series, errors="coerce")
    bad = values.isna() | (values % 1 != 0) | (values < minimum)
    if bad.any():
        raise DataError(f"{name}: {int(bad.sum())} valores enteros inválidos.")
    if len(values) and int(values.max()) > 2**63 - 1:
        raise DataError(f"{name}: valores fuera del rango int64.")
    try:
        return values.astype("int64")
    except (ValueError, OverflowError, TypeError) as exc:
        raise DataError(f"{name}: valores fuera del rango int64.") from exc


def datetime_column(series, name):
    """Aceptar ISO y día/mes/año explícitos; conservar tiempos sin inventar zona horaria."""
    source = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y %H:%M:%S",
                "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M"):
        missing = parsed.isna()
        parsed.loc[missing] = pd.to_datetime(source.loc[missing], format=fmt, errors="coerce")
    if parsed.isna().any():
        raise DataError(f"{name}: {int(parsed.isna().sum())} fechas inválidas o sin formato soportado.")
    return parsed


def normalize_trips(frame, month):
    """Construir las cinco columnas operativas; excluir campos personales de la salida."""
    mapping = column_mapping(frame.columns, TRIP_SCHEMA)
    clean = frame[list(mapping)].rename(columns=mapping).copy()
    for name in ("trip_id", "origin_station_id", "destination_station_id"):
        clean[name] = integer_column(clean[name], name, minimum=1)
    for name in ("started_at", "ended_at"):
        clean[name] = datetime_column(clean[name], name)
    if (clean.ended_at < clean.started_at).any():
        raise DataError("Hay viajes cuyo fin es anterior al inicio; no se descartan silenciosamente.")
    if (clean.started_at.dt.strftime("%Y-%m") != month).any():
        raise DataError("Hay inicios de viaje fuera del mes solicitado.")
    return clean[TRIP_SCHEMA.names]


def normalize_stations(frame):
    """Validar catálogo completo; no inferir municipio ni eliminar estaciones históricas."""
    mapping = column_mapping(frame.columns, STATION_SCHEMA)
    clean = frame[list(mapping)].rename(columns=mapping).copy()
    clean["station_id"] = integer_column(clean.station_id, "station_id", minimum=1)
    clean["capacity"] = integer_column(clean.capacity, "capacity")
    if clean.station_id.duplicated().any():
        raise DataError("El catálogo tiene identificadores de estación duplicados.")
    for name, low, high in (("latitude", -90, 90), ("longitude", -180, 180)):
        clean[name] = pd.to_numeric(clean[name], errors="coerce")
        if not clean[name].between(low, high).all():
            raise DataError(f"{name}: coordenadas inválidas.")
    clean["station_name"] = clean.station_name.astype("string").str.normalize("NFC").str.strip()
    if clean.station_name.isna().any() or clean.station_name.eq("").any():
        raise DataError("Hay nombres de estación vacíos.")
    return clean[STATION_SCHEMA.names]


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.append(dict(attrs).get("href", ""))


def source_urls(html, months, station_filename):
    """Encontrar rutas reales del portal: el directorio de cada mes no es constante."""
    parser = LinkParser()
    parser.feed(html)
    wanted = [f"datos_abiertos_{m.replace('-', '_')}.csv" for m in months]
    wanted.append(station_filename)
    found = {}
    for href in parser.hrefs:
        url = urljoin(config.SOURCE_PAGE, href)
        parsed = urlparse(url)
        name = Path(parsed.path).name
        if name in wanted and parsed.scheme == "https" and parsed.hostname in (
                "mibici.net", "www.mibici.net"):
            if name in found and found[name] != url:
                raise DataError(f"Hay enlaces distintos para {name}.")
            found[name] = url
    missing = set(wanted) - found.keys()
    if missing:
        raise DataError(f"No están publicados los archivos solicitados: {sorted(missing)}.")
    return found


def http_session():
    retry = Retry(total=config.HTTP_RETRIES, backoff_factor=1,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    session = requests.Session()
    session.headers["User-Agent"] = "mibici-equipo13/0.1"
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download(session, url, destination, expected_hash=None):
    """Descargar atómicamente o verificar caché; jamás aceptar un hash esperado distinto."""
    destination = Path(destination)
    if expected_hash and destination.exists():
        if sha256_file(destination) != expected_hash:
            raise DataError(f"El archivo local cambió: {destination.name}.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as out:
            temporary = Path(out.name)
            with session.get(url, stream=True, timeout=config.TIMEOUT_SECONDS) as response:
                response.raise_for_status()
                for block in response.iter_content(1024 * 1024):
                    out.write(block)
        if expected_hash and sha256_file(temporary) != expected_hash:
            raise DataError(f"La fuente cambió: {destination.name}; SHA-256 no coincide.")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parquet_atomic(frames, schema, destination):
    """Escribir por lotes y reemplazar salida solo cuando todos los lotes son válidos."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with pq.ParquetWriter(temporary, schema, compression="zstd", version="2.6") as writer:
            for frame in frames:
                writer.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def file_record(path, data_dir, **metadata):
    return {"path": path.relative_to(data_dir).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256_file(path), **metadata}


def process_sources(session, sources, data_dir, months, chunk_size, expected=None):
    """Procesar catálogo y meses sin cargar el conjunto completo de viajes en memoria."""
    raw_expected = {f["path"]: f["sha256"] for f in (expected or {}).get("files", [])}
    files = []
    station_source = sources["stations"]
    station_raw = data_dir / "external" / Path(urlparse(station_source).path).name
    download(session, station_source, station_raw,
             raw_expected.get(station_raw.relative_to(data_dir).as_posix()))
    encoding, delimiter = csv_format(station_raw)
    station_frame = pd.read_csv(station_raw, encoding=encoding, sep=delimiter, dtype="string")
    stations = normalize_stations(station_frame)
    station_ids = set(stations.station_id)
    station_output = data_dir / "processed" / "stations.parquet"
    parquet_atomic([stations], STATION_SCHEMA, station_output)
    files.extend([
        file_record(station_raw, data_dir, kind="raw_stations", url=station_source,
                    encoding=encoding, delimiter=delimiter, rows=len(stations),
                    source_columns=station_frame.columns.tolist()),
        file_record(station_output, data_dir, kind="stations", rows=len(stations),
                    schema=str(STATION_SCHEMA)),
    ])
    total_rows, total_raw_rows, duplicates_removed = 0, 0, 0
    all_station_ids, missing_station_ids = set(), set()
    for month in months:
        source = sources[month]
        raw = data_dir / "raw" / Path(urlparse(source).path).name
        download(session, source, raw, raw_expected.get(raw.relative_to(data_dir).as_posix()))
        encoding, delimiter = csv_format(raw)
        headers = pd.read_csv(raw, encoding=encoding, sep=delimiter, nrows=0).columns.tolist()
        mapping = column_mapping(headers, TRIP_SCHEMA)
        output = data_dir / "processed" / f"trips_{month}.parquet"
        stats = {"rows": 0, "raw_rows": 0, "duplicate_rows_removed": 0,
                 "min_started_at": None, "max_started_at": None}
        seen_trip_ids = {}

        def frames():
            reader = pd.read_csv(raw, encoding=encoding, sep=delimiter, dtype="string",
                                 usecols=list(mapping), chunksize=chunk_size, on_bad_lines="error")
            for frame in reader:
                clean = normalize_trips(frame, month)
                stats["raw_rows"] += len(clean)
                keep = []
                for row in clean.itertuples(index=False, name=None):
                    trip_id, signature = row[0], row[1:]
                    if trip_id in seen_trip_ids:
                        if seen_trip_ids[trip_id] != signature:
                            raise DataError(f"{month}: identificador de viaje con datos contradictorios.")
                        keep.append(False)
                        stats["duplicate_rows_removed"] += 1
                    else:
                        seen_trip_ids[trip_id] = signature
                        keep.append(True)
                clean = clean.loc[keep]
                stats["rows"] += len(clean)
                if len(clean):
                    low, high = clean.started_at.min().isoformat(), clean.started_at.max().isoformat()
                    stats["min_started_at"] = min(stats["min_started_at"] or low, low)
                    stats["max_started_at"] = max(stats["max_started_at"] or high, high)
                used = set(clean.origin_station_id) | set(clean.destination_station_id)
                all_station_ids.update(used)
                missing_station_ids.update(used - station_ids)
                yield clean

        parquet_atomic(frames(), TRIP_SCHEMA, output)
        if not stats["rows"]:
            raise DataError(f"{month}: el archivo no contiene viajes.")
        total_rows += stats["rows"]
        total_raw_rows += stats["raw_rows"]
        duplicates_removed += stats["duplicate_rows_removed"]
        files.extend([
            file_record(raw, data_dir, kind="raw_trips", month=month, url=source,
                        encoding=encoding, delimiter=delimiter, rows=stats["raw_rows"],
                        source_columns=headers),
            file_record(output, data_dir, kind="trips", month=month, schema=str(TRIP_SCHEMA), **stats),
        ])
    return files, {"months": len(months), "raw_trips": total_raw_rows, "trips": total_rows,
                   "duplicate_rows_removed": duplicates_removed, "catalog_stations": len(stations),
                   "trip_stations": len(all_station_ids),
                   "station_ids_missing_from_catalog": sorted(missing_station_ids)}


def environment_versions():
    return {"pandas": pd.__version__, "pyarrow": pa.__version__}


def build_manifest(session, sources, data_dir, months, chunk_size, expected=None):
    files, summary = process_sources(session, sources, data_dir, months, chunk_size, expected)
    manifest = {"manifest_version": config.MANIFEST_VERSION,
                "transform_version": config.TRANSFORM_VERSION,
                "source_page": config.SOURCE_PAGE, "months": months,
                "chunk_size": chunk_size, "versions": environment_versions(),
                "time_policy": "source_naive_no_timezone_conversion",
                "scope": "all_stations_no_municipality_filter",
                "files": files, "summary": summary}
    manifest["data_version"] = data_version(manifest)
    return manifest


def manifest_path(data_dir, relative):
    """Impedir que rutas de un manifiesto escapen del directorio local de datos."""
    target = (data_dir / relative).resolve()
    if not target.is_relative_to(data_dir.resolve()):
        raise DataError("El manifiesto contiene una ruta fuera del directorio de datos.")
    return target


def read_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != config.MANIFEST_VERSION:
        raise DataError("Versión de manifiesto no soportada.")
    if manifest.get("data_version") != data_version(manifest):
        raise DataError("El manifiesto fue modificado: data_version no coincide.")
    return manifest


def verify_manifest(manifest, data_dir):
    """Verificar integridad y filas de CSV/Parquet sin red ni modificar datos."""
    for entry in manifest["files"]:
        path = manifest_path(data_dir, entry["path"])
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise DataError(f"Archivo ausente o modificado: {entry['path']}.")
        if path.stat().st_size != entry["bytes"]:
            raise DataError(f"Tamaño distinto: {entry['path']}.")
        if path.suffix == ".parquet":
            rows = pq.read_metadata(path).num_rows
        else:
            with path.open(encoding=entry["encoding"], newline="") as stream:
                rows = sum(1 for _ in csv.reader(stream, delimiter=entry["delimiter"])) - 1
        if rows != entry["rows"]:
            raise DataError(f"Número de registros distinto: {entry['path']}.")


def save_manifest(manifest, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                     suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(manifest, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("ingest", "verify", "reproduce"))
    parser.add_argument("--start-month", default=config.DEFAULT_START_MONTH)
    parser.add_argument("--end-month", default=config.DEFAULT_END_MONTH)
    parser.add_argument("--stations-file", default=config.STATIONS_FILENAME)
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=config.DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    try:
        if args.chunk_size < 1:
            raise DataError("chunk-size debe ser positivo.")
        if args.command == "verify":
            manifest = read_manifest(args.manifest)
            verify_manifest(manifest, args.data_dir)
        elif args.command == "reproduce":
            expected = read_manifest(args.manifest)
            if expected["transform_version"] != config.TRANSFORM_VERSION:
                raise DataError("Se requiere la versión original del código de transformación.")
            if expected["versions"] != environment_versions():
                raise DataError("Instala las versiones de pandas/pyarrow del manifiesto.")
            sources = {f.get("month", "stations"): f["url"] for f in expected["files"]
                       if f["kind"].startswith("raw_")}
            with http_session() as session:
                manifest = build_manifest(session, sources, args.data_dir, expected["months"],
                                          expected["chunk_size"], expected)
            if manifest != expected:
                raise DataError("La reconstrucción difiere del manifiesto; no se cambió su versión.")
            verify_manifest(expected, args.data_dir)
        else:
            if args.manifest.exists():
                raise DataError("El manifiesto ya existe: usa reproduce/verify o una ruta nueva.")
            months = month_range(args.start_month, args.end_month)
            with http_session() as session:
                response = session.get(config.SOURCE_PAGE, timeout=config.TIMEOUT_SECONDS)
                response.raise_for_status()
                found = source_urls(response.text, months, args.stations_file)
                sources = {m: found[f"datos_abiertos_{m.replace('-', '_')}.csv"] for m in months}
                sources["stations"] = found[args.stations_file]
                manifest = build_manifest(session, sources, args.data_dir, months, args.chunk_size)
            verify_manifest(manifest, args.data_dir)
            save_manifest(manifest, args.manifest)
    except (DataError, requests.RequestException, OSError, ValueError, pa.ArrowException) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps({"command": args.command, "data_version": manifest["data_version"],
                      "summary": manifest["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
