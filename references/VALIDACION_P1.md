# Validación de P1 — 17 de octubre de 2026

Entorno comprobado: macOS ARM64, Python 3.12.15 y dependencias fijadas en requirements.txt. Windows y Linux no se ejecutaron.

Se creó un clon local limpio del commit de implementación y documentación `abbed01a487be786e0068c711ea9a6b148fd64d9`, con entorno y tracking independientes. Se instaló siguiendo README. Para evitar repetir la transferencia, se reutilizaron únicamente los CSV originales y GeoJSON ya descargados, comprobados por SHA-256; todos los Parquet se reconstruyeron en el clon.

| Comprobación | Resultado |
|---|---|
| Instalación de dependencias y paquete editable | Correcta. |
| `python -m pip check` | Sin incompatibilidades. |
| Reconstrucción del manifiesto de 24 meses | Hashes y conteos idénticos. |
| Geometría oficial fijada por hash | Correcta. |
| `python -m pytest -q -W error::DeprecationWarning` | 37 pruebas aprobadas. |
| Tres baselines ejecutados en tracking nuevo | Todas las métricas, particiones y selección idénticas. |
| Código y dependencias después de ejecutar | Sin modificaciones. |

Los runIDs y el commit del clon corresponden a esa nueva ejecución; los resultados versionados documentan la ejecución original del commit `49f8c9ac1422a54a4a2fdc8e24493fc2c10cc350`. La reproducción crea sus propios identificadores y actualiza los archivos de resultados, sin alterar el protocolo.

Versión de datos: `8be1e590639cdb6ad5e05817bec67b1d0a3b1d792fea2215dd3fec1a19309030`.

Se verificó por lectura nativa y PDF exportado la sección de P1 del documento compartido: tres tablas nativas, figura de arquitectura sin recortes y conservación del texto P2/P3. Los cambios posteriores al commit validado corrigen únicamente la flecha de salida de reportes del diagrama y añaden documentación/evidencias; no modifican el cálculo de P1.

## Evidencia MLflow

Capturas reales de la interfaz local, en `reports/figures/evidence/p1/`:

1. `01-runs.png`: lista de tres runs del periodo definitivo.
2. `02-comparison.png`: comparación de esos runs.
3. `03-run-detail.png`: parámetros, métricas y etiquetas del run de persistencia.
4. `04-artifact.png`: figura de prueba cargada desde los artefactos del run.

Las capturas cubren P1. La evidencia global del proyecto debe actualizarse cuando los responsables incorporen P2 y P3.
