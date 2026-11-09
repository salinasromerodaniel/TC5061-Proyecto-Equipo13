"""La entrada histórica de CI conserva los datos y rechaza cambios de contenido."""

import base64
import gzip
import hashlib
from pathlib import Path
import runpy

import pytest

from mibici import config

restore_catalog = runpy.run_path(
    str(config.PROJ_ROOT / ".github/scripts/restore_catalog.py"))["restore_catalog"]


def snapshot(content=b"id,name\n1,synthetic station\n"):
    archive = base64.b64encode(gzip.compress(content, mtime=0)).decode("ascii")
    manifest = {"files": [{"kind": "raw_stations", "bytes": len(content),
                           "sha256": hashlib.sha256(content).hexdigest()}]}
    return content, archive, manifest


def test_restored_catalog_is_identical_and_idempotent(tmp_path):
    content, archive, manifest = snapshot()
    output = tmp_path / "cache/catalog.csv"
    assert restore_catalog(archive, manifest, output).read_bytes() == content
    assert restore_catalog(archive, manifest, output).read_bytes() == content


@pytest.mark.parametrize("alteration", ["digest", "size", "content"])
def test_modified_snapshot_is_rejected_without_replacing_previous_file(tmp_path, alteration):
    content, archive, manifest = snapshot()
    if alteration == "digest":
        manifest["files"][0]["sha256"] = "0" * 64
    elif alteration == "size":
        manifest["files"][0]["bytes"] -= 1
    else:
        _, archive, _ = snapshot(content.replace(b"synthetic", b"changed!!"))
    output = tmp_path / "catalog.csv"
    output.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="no coincide"):
        restore_catalog(archive, manifest, output)
    assert output.read_bytes() == b"preserve"


@pytest.mark.parametrize("archive", [None, "", "invalid-base64"])
def test_missing_or_invalid_archive_cannot_create_a_catalog(tmp_path, archive):
    _, _, manifest = snapshot()
    output = tmp_path / "catalog.csv"
    with pytest.raises(ValueError):
        restore_catalog(archive, manifest, output)
    assert not output.exists()


def test_different_existing_catalog_is_not_overwritten(tmp_path):
    _, archive, manifest = snapshot()
    output = tmp_path / "catalog.csv"
    output.write_bytes(b"another catalog")
    with pytest.raises(ValueError, match="contenido diferente"):
        restore_catalog(archive, manifest, output)
    assert output.read_bytes() == b"another catalog"
