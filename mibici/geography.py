"""Selección espacial reproducible con geometría oficial del INEGI."""

import json
from pathlib import Path

from shapely.geometry import Point, shape

from mibici import config
from mibici.dataset import DataError, download, http_session, save_manifest, sha256_file


def boundary_geometry(document):
    """Validar que la respuesta contenga únicamente el municipio solicitado."""
    features = document.get("features", [])
    if document.get("type") != "FeatureCollection" or len(features) != 1:
        raise DataError("Se esperaba una sola geometría municipal de INEGI.")
    feature = features[0]
    properties = feature["properties"]
    if (properties.get("cvegeo") != config.P1_MUNICIPALITY_CODE
            or properties.get("nomgeo") != config.P1_MUNICIPALITY_NAME):
        raise DataError("La geometría no corresponde a Zapopan (14120).")
    geometry = shape(feature["geometry"])
    if geometry.is_empty or not geometry.is_valid or geometry.geom_type not in ("Polygon", "MultiPolygon"):
        raise DataError("El polígono municipal no es válido.")
    return geometry


def load_boundary(data_dir=config.DATA_DIR, manifest_path=config.P1_BOUNDARY_MANIFEST):
    """Descargar y verificar la geometría fijada en el manifiesto geográfico."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest["municipality_code"] != config.P1_MUNICIPALITY_CODE:
        raise DataError("El manifiesto geográfico no corresponde a Zapopan.")
    path = Path(data_dir) / "external" / "zapopan-inegi.geojson"
    with http_session() as session:
        download(session, manifest["url"], path, manifest["sha256"])
    return boundary_geometry(json.loads(path.read_text(encoding="utf-8"))), manifest


def municipality_station_ids(stations, geometry):
    """Seleccionar coordenadas cubiertas por el polígono, incluido su borde."""
    return sorted(int(row.station_id) for row in stations.itertuples(index=False)
                  if geometry.covers(Point(float(row.longitude), float(row.latitude))))


def main():
    path = config.DATA_DIR / "external" / "zapopan-inegi.geojson"
    if config.P1_BOUNDARY_MANIFEST.exists():
        _, manifest = load_boundary()
    else:
        with http_session() as session:
            download(session, config.P1_BOUNDARY_URL, path)
        document = json.loads(path.read_text(encoding="utf-8"))
        boundary_geometry(document)
        manifest = {"source": "INEGI, Servicio Web del Catálogo Único de Claves Geoestadísticas",
                    "documentation_url": "https://www.inegi.org.mx/servicios/catalogounico.html",
                    "url": config.P1_BOUNDARY_URL,
                    "municipality_code": config.P1_MUNICIPALITY_CODE,
                    "municipality_name": config.P1_MUNICIPALITY_NAME,
                    "sha256": sha256_file(path), "bytes": path.stat().st_size,
                    "geometry_type": document["features"][0]["geometry"]["type"],
                    "coordinate_order": "longitude_latitude", "selection": "polygon_covers_point",
                    "temporal_policy": "reference_snapshot_not_historical_boundaries"}
        save_manifest(manifest, config.P1_BOUNDARY_MANIFEST)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
