# Validación del bloque de ingesta

Fecha: 17 de octubre de 2026. Sistema probado: macOS ARM64, Python 3.12.15. Windows y Linux todavía no se han ejecutado.

Se clonó el commit de implementación en otro directorio, sin datos ni entorno previos. Se creó un entorno nuevo y se siguieron los comandos de instalación del README con pip, incluyendo el paquete local editable.

| Comprobación | Resultado |
|---|---|
| `python -m pip check` | Sin dependencias incompatibles. |
| `python -m pytest -q` | 24 pruebas aprobadas. |
| `python -m mibici.dataset reproduce --manifest references/data-manifest-2026-09.json` | Descarga y reconstrucción con hashes idénticos. |
| `python -m mibici.dataset verify --manifest references/data-manifest-2026-09.json` | Hashes, tamaños y conteos coinciden, sin red. |
| Estado del clon después de ejecutar | Sin cambios en archivos versionados. |

Versión de datos: `c76da3761993210d5131ea70a7ae57f78ac5e9dad3a6a0d1c243915d4d3801d4`.

Resultado: 490,056 filas originales; 758 duplicados operativos removidos; 489,298 viajes limpios; 468 estaciones usadas; 484 entradas del catálogo; ninguna referencia de estación ausente.

Las pruebas sintéticas cubren esquemas ambiguos, campos personales excluidos, formatos de fecha, datos inválidos, duplicados entre lotes, conflictos del mismo ID, catálogo histórico, codificación CP1252, descubrimiento de enlaces, reproducción determinista, alteraciones de manifiesto/archivos y fallos de hash durante descarga.

Esta validación cubre el bloque de ingesta con un mes; no acredita todavía millones de registros, selección del periodo final, partición temporal, filtro municipal, variables, modelos ni experimentos MLflow.
