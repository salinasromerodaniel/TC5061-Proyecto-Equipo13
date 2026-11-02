# Validación de datos P1, P2 y P3 con Deepchecks

## Alcance y contratos

Solo se añade la capa de calidad de datos; no se entrena ni modifica ningún modelo. Se reutilizan las funciones reales del proyecto en `mibici/features.py`, `mibici/p2_panel.py` y las agregaciones de P2. El comando de equipo es `mibici.validation_tasks`; `mibici.validation` conserva la interfaz anterior de P2.

| Pregunta | Unidad y datos comprobados | Integridad | Drift |
|---|---|---|---|
| P1, regresión | Estación-semana objetivo; `lag_1`, `mean_previous_4`, `lag_52`, `week_sin`, `week_cos`; objetivo: salidas observadas | Tipos, lunes, bloques temporales ordenados, conteos no negativos, objetivo entero, calendario en rango, faltantes y clave estación-semana única | Features y etiqueta continua, train/validation y train/test |
| P2, clasificación | Panel real de Zapopan, siete predictores y objetivo binario de déficit neto la semana siguiente | Tipos, horizonte de siete días, lunes, bloques temporales ordenados, capacidad positiva, faltantes y clave estación-semana única | Features y etiqueta binaria, train/validation y train/test |
| P3, clustering | Estación; `departures`, `arrivals`, `traffic`, `balance`, `capacity`, antes de escalar | Tipos, estación única, faltantes, finitos, no negativos donde corresponde; `traffic = departures + arrivals`, `balance = departures - arrivals` | No aplica: el pipeline actual agrega todo el periodo y no define train/test |

El contrato previo rechaza columnas inesperadas, tipos incorrectos e infinitos. Así se evita incluir columnas de usuarios en los reportes. Deepchecks ejecuta `PercentOfNulls` y `DataDuplicates` sobre todas las filas. Los duplicados se definen por entidad y periodo, no por repetición de los valores de features.

P1 permite nulos únicamente en `lag_52`: faltan durante el primer año de historia. El candidato HistGradientBoosting los admite; el baseline estacional recurre a `lag_1`. Se informa su cantidad y un chequeo de faltantes sin condición de rechazo para esa columna. Si una comparación tiene la columna completamente vacía en alguno de los dos bloques, se excluye solo esa variable del drift y se deja constancia en `drift_excluded_all_missing`. Los nulos de las demás columnas sí incumplen integridad. No se imputa el panel para conseguir un PASS.

P3 usa exactamente el alcance del código del equipo: catálogo completo, sin aplicar el filtro municipal de P1/P2. Su función original ya sustituye nulos por cero antes de devolver la matriz. Por ello, estos chequeos acreditan la matriz transformada, no la ausencia de faltantes en las fuentes originales. Se permite capacidad cero y balance negativo, tal como permite su pipeline. No se inventan etiquetas de clase ni una partición aleatoria para simular drift.

P2 usa un indicador indirecto: los chequeos no convierten el objetivo en observaciones reales de estaciones vacías, inventario ni stockouts.

## Instalación en Windows

Desde la raíz del repositorio, con Python 3.12:

```powershell
py -3.12 -m venv .venv-deepchecks
.\.venv-deepchecks\Scripts\python.exe -m pip install -r requirements-deepchecks.txt
.\.venv-deepchecks\Scripts\python.exe -m pip check
```

Este entorno es separado del de entrenamiento: usa NumPy 1.26.4 y scikit-learn 1.5.2. No combinarlo con `requirements.txt`, que requiere NumPy 2. PyArrow y Shapely permiten exportar desde los Parquet de Windows. No hay que activar el entorno ni instalar el proyecto con `-e .`.

Deepchecks 0.19.1 importa `pkg_resources`; por eso se fija `setuptools==80.9.0`. A partir de setuptools 82 ese módulo fue eliminado. Un `pip check` exitoso no sustituye una prueba de importación/integración. Los avisos de deprecación de estas versiones se conservan y no equivalen a errores de calidad del dataset.

## Pruebas y demo

Un comando ejecuta las pruebas de esta contribución, incluidas las suites reales de Deepchecks con datos sintéticos:

```powershell
.\.venv-deepchecks\Scripts\python.exe -m unittest discover -s tests -p test_validation.py -v
```

También son descubribles por PyTest. Si Deepchecks no está instalado, se omiten las pruebas de integración; eso no acredita su funcionamiento.

Para generar los reportes de demostración de las tres preguntas:

```powershell
.\.venv-deepchecks\Scripts\python.exe -m mibici.validation_tasks check --demo
```

La demo no acredita calidad de MiBici. Sus cambios de calendario y distribución pueden producir `DRIFT_REQUIRES_REVIEW`.

## Ejecución con los datos reales

Requiere los Parquet originales de la ingesta y los manifiestos versionados. Si falta el polígono local, prepararlo explícitamente mediante el flujo existente (usa red y verifica el SHA-256 fijado):

```powershell
.\.venv-deepchecks\Scripts\python.exe -m mibici.geography
```

Exportar las tres matrices reales con un comando:

```powershell
.\.venv-deepchecks\Scripts\python.exe -m mibici.validation_tasks export
```

La exportación verifica el manifiesto de datos y sus archivos antes de reutilizar las funciones del proyecto. Guarda `data/processed/quality/p1_panel.csv`, `p2_panel.csv`, `p3_panel.csv` y sus archivos `.metadata.json`, con SHA-256 y procedencia. No entrena ni requiere MLflow o un commit limpio. El validador no descarga silenciosamente fuentes. Si alguna pregunta falla, lo registra y sale con error aunque las otras se exporten.

Ejecutar los chequeos de las tres preguntas:

```powershell
.\.venv-deepchecks\Scripts\python.exe -m mibici.validation_tasks check
```

Para seleccionar una pregunta añadir `--question P1`, `P2` o `P3`. `--manifest`, `--data-dir`, `--panel-dir` y `--report-dir` permiten especificar las rutas. No editar manualmente los CSV ni sus hashes para ocultar errores: volver a exportarlos desde el pipeline.

## Resultados y criterios

Cada ejecución crea una carpeta UTC nueva en `reports/deepchecks/real_*` (o `demo_*`/`export_*`). El resumen raíz contiene cada pregunta; los detalles están en subcarpetas P1, P2 y P3. P1/P2 generan cinco HTML cada una: integridad de tres bloques y dos comparaciones de drift. P3 genera un HTML de integridad. Se guardan también los JSON de las suites, versiones de librerías, faltantes y procedencia.

Los HTML se solicitan con `connected=False`. Se desactiva la consulta automática de versión de Deepchecks. No se usa Deepchecks Cloud ni se suben datos a un servicio externo.

| Estado/código | Interpretación |
|---|---|
| PASS / 0 | Integridad aprobada y drift dentro del umbral, si aplica |
| DRIFT_REQUIRES_REVIEW / 0 | Integridad aprobada; drift fuera del umbral exploratorio |
| SCHEMA_FAILED o INTEGRITY_FAILED / 1 | Contrato o calidad incumplidos |
| Drift / 2 con `--strict-drift` | La opción convierte la alerta de drift en fallo |
| ERROR o CHECK_ERROR / 3 | Error de ejecución, procedencia o chequeo no ejecutado |

No se consideran aprobados los chequeos que no llegaron a ejecutarse. El comando de equipo devuelve error si falla cualquier pregunta. `--strict-drift` afecta P1/P2; P3 no tiene drift definido.

Se usa semilla 13 y umbral exploratorio 0.2, centralizado en `config.py`; no es un estándar universal. Features numéricas usan KS y las etiquetas el método correspondiente de Deepchecks. Sus puntuaciones no son intercambiables. Una alerta requiere analizar composición de cohortes, calendario y cambios de actividad; no permite ajustar el test ni eliminar filas para mejorar las métricas.

## Evidencia para entregar

1. Captura de las pruebas terminadas, con número de casos y `OK`.
2. Captura del comando real que muestre estado de P1, P2 y P3 y ubicación de reportes.
3. Capturas de los reportes de integridad y de las alertas reales de drift de P1/P2, con interpretación. Para P3 mostrar su integridad y explicar por qué no hay train/test.

Conservar los reportes completos localmente, fuera de Git. Añadir al entregable solo evidencia revisada sin datos personales y una síntesis de resultados. Los datos, el entorno adicional y los reportes de Deepchecks están excluidos de Git.

## Referencias

Deepchecks. (s. f.). *DataDuplicates*. https://docs.deepchecks.com/stable/api/generated/deepchecks.tabular.checks.data_integrity.DataDuplicates.html

Deepchecks. (s. f.). *FeatureDrift*. https://docs.deepchecks.com/stable/api/generated/deepchecks.tabular.checks.train_test_validation.FeatureDrift.html

Deepchecks. (s. f.). *LabelDrift*. https://docs.deepchecks.com/stable/api/generated/deepchecks.tabular.checks.train_test_validation.LabelDrift.html

Deepchecks. (s. f.). *PercentOfNulls*. https://docs.deepchecks.com/stable/api/generated/deepchecks.tabular.checks.data_integrity.PercentOfNulls.html

Python Packaging Authority. (s. f.). *Package discovery and resource access using pkg_resources*. https://setuptools.pypa.io/en/stable/deprecated/pkg_resources.html
