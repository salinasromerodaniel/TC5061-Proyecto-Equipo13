"""Restaurar para CI el catálogo original, verificándolo contra el manifiesto."""

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def restore_catalog(archive, manifest, destination):
    if not archive:
        raise ValueError("Falta MIBICI_CATALOG_ARCHIVE; configura la copia histórica de CI.")
    records = [entry for entry in manifest["files"] if entry["kind"] == "raw_stations"]
    if len(records) != 1:
        raise ValueError("El manifiesto debe contener un catálogo original.")
    record = records[0]
    compressed = base64.b64decode(archive.strip(), validate=True)
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        content = stream.read(record["bytes"] + 1)
    if (len(content) != record["bytes"]
            or hashlib.sha256(content).hexdigest() != record["sha256"]):
        raise ValueError("El catálogo no coincide con el tamaño y SHA-256 del manifiesto.")
    destination = Path(destination)
    if destination.exists():
        if destination.read_bytes() != content:
            raise ValueError("El destino ya existe con un contenido diferente.")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_bytes(content)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "references/data-manifest-2024-10_2026-09-v2.json")
    archive_choice = parser.add_mutually_exclusive_group()
    archive_choice.add_argument("--archive-file", type=Path)
    archive_choice.add_argument("--pack-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pack_source:
        if args.pack_source.resolve() == args.output.resolve():
            raise ValueError("El archivo original y la salida deben ser distintos.")
        content = args.pack_source.read_bytes()
        archive = base64.b64encode(gzip.compress(content, mtime=0)).decode("ascii")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            restore_catalog(archive, manifest, Path(temporary) / "verified.csv")
        args.output.write_text(archive, encoding="ascii")
        print(f"Snapshot de CI creada y verificada: {args.output}")
        return
    archive = (args.archive_file.read_text(encoding="ascii") if args.archive_file
               else os.getenv("MIBICI_CATALOG_ARCHIVE"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    path = restore_catalog(archive, manifest, args.output).resolve()
    if os.getenv("GITHUB_ENV"):
        with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as stream:
            stream.write(f"MIBICI_CATALOG_SOURCE={path}\n")
    print(f"Catálogo histórico restaurado y verificado: {path}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, EOFError) as error:
        raise SystemExit(f"Error: {error}") from error
