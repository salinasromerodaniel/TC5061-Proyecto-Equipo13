# Catálogo histórico en CI

La URL original de septiembre de 2026 dejó de estar disponible en MiBici.
Para reproducir los mismos datos, CI utiliza una copia del archivo original
almacenada como entrada protegida de GitHub Actions, fuera del historial de Git.
No usa el catálogo de otro mes.

La integración sigue habilitada en Windows, Linux y macOS. El workflow:

1. Restaura la copia gzip/base64 desde `MIBICI_CATALOG_2026_09_GZIP_B64`.
2. Comprueba tamaño y SHA-256 contra el manifiesto de los 24 meses.
3. Publica la ruta temporal como `MIBICI_CATALOG_SOURCE`.
4. Ejecuta `test_published_station_catalog_bytes`: vuelve a comprobar el hash
   del CSV, genera el Parquet y exige los mismos bytes y hash publicados.

Una entrada ausente, corrupta o diferente hace fallar el proceso. No se omite
la prueba ni se cambian los manifiestos. La copia contiene únicamente el catálogo
público de estaciones; se mantiene protegida para que su contenido no aparezca
en los logs. No se suben CSV ni archivos comprimidos a Git.

## Configuración del repositorio

El repositorio del equipo ya tiene configurada esa entrada. Si debe restaurarse,
un administrador con una copia original puede preparar el archivo con el entorno
del proyecto activado.

macOS/Linux, desde la raíz del repositorio:

```bash
python .github/scripts/restore_catalog.py --pack-source data/external/nomenclatura_2026_09.csv --output /tmp/mibici-catalog-2026-09.gz.b64
gh secret set MIBICI_CATALOG_2026_09_GZIP_B64 --repo salinasromerodaniel/TC5061-Proyecto-Equipo13 < /tmp/mibici-catalog-2026-09.gz.b64
```

PowerShell:

```powershell
$archive = Join-Path $env:TEMP "mibici-catalog-2026-09.gz.b64"
python .github/scripts/restore_catalog.py --pack-source data/external/nomenclatura_2026_09.csv --output $archive
Get-Content -Raw $archive | gh secret set MIBICI_CATALOG_2026_09_GZIP_B64 --repo salinasromerodaniel/TC5061-Proyecto-Equipo13
```

La preparación rechaza una copia que no coincida con el manifiesto. El SHA-256
original es `206e45fe633922b634d858142fa21e6fd9f4c421b6435c02980ef290cc2df90d`;
el catálogo tiene 40,841 bytes.

## Ejecución local

Con la copia original en `data/external/`, se puede activar la integración
sin descargarla del portal. En macOS/Linux:

```bash
MIBICI_CATALOG_INTEGRATION=1 MIBICI_CATALOG_SOURCE=data/external/nomenclatura_2026_09.csv python -m pytest tests/test_dataset.py -q -W error::DeprecationWarning
```

En PowerShell:

```powershell
$env:MIBICI_CATALOG_INTEGRATION = "1"
$env:MIBICI_CATALOG_SOURCE = (Resolve-Path data/external/nomenclatura_2026_09.csv).Path
python -m pytest tests/test_dataset.py -q -W error::DeprecationWarning
```

Si no se define `MIBICI_CATALOG_SOURCE`, la prueba conserva la descarga desde
la URL del manifiesto. La disponibilidad de esa URL es distinta de la
verificación de integridad del catálogo conservado.
