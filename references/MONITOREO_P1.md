# Monitoreo P1: referencia congelada y Evidently

Comparamos el entrenamiento de P1 con octubre de 2026, el primer mes completo posterior al
histórico. El protocolo queda fijado en [monitoring-p1-protocol.json](monitoring-p1-protocol.json).
Conservamos los datos, modelos, pruebas y API de las entregas anteriores.

## Fuentes y cortes

| Partición | Inicio incluido | Fin excluido | Filas | Estaciones | Uso |
|---|---|---|---:|---:|---|
| reference | 2024-11-04 | 2026-06-08 | 5927 | 73 | Entrenamiento original de A y referencia del drift |
| historical_validation | 2026-06-08 | 2026-08-03 | 609 | 78 | Ajuste histórico de B |
| historical_test | 2026-08-03 | 2026-09-28 | 624 | 78 | Evaluación histórica ya utilizada; conservar como antecedente |
| current | 2026-10-05 | 2026-10-26 | 234 | 78 | Reserva completa para comparar A y B |

A corresponde a `mibici-equipo13-p1-demanda/1`, HGB Poisson con siete hojas,
run `65ed8a9510984cf2aedde1663050440a`. Su entrenamiento utiliza el manifiesto
histórico de 24 meses, versión `6622c0fe…`. Reconstruimos las 5927 filas originales;
el digest de entrenamiento MLflow local coincide con `b1788037`. La identidad de A
se fija por nombre, versión, run, commit, versión de datos y SHA del modelo:
una versión 1 de otro Registry no identifica necesariamente el mismo artefacto.

El manifiesto de octubre fija 635532 viajes sin duplicados eliminados, versión
`da76e200…`, y sus URLs/hashes. El catálogo de octubre reporta una estación usada por
viajes sin ficha (124); queda fuera de la cohorte P1. Conservamos la clasificación
geográfica y la cohorte históricas, sin recalcularlas con el catálogo nuevo.

Las semanas comienzan el lunes, con hora local tal como se publica. Excluimos
semanas incompletas: septiembre28–octubre4 sólo aporta historia, octubre26–noviembre1
queda fuera del objetivo actual. Sumamos ambos meses para completar la primera.
Los retardos se desplazan antes del objetivo; el protocolo es de origen móvil semanal:
para predecir el12 de octubre ya conocemos las salidas observadas del5.
No representa un pronóstico simultáneo de todo octubre.

## Reproducción

Usamos Python3.12. Desde la raíz del repositorio:

```bash
# Entorno principal existente y datos históricos preparados con su manifiesto.
python -m mibici.dataset verify --manifest references/data-manifest-2024-10_2026-09-v2.json

# Octubre se reconstruye aparte; no sustituye data/processed/stations.parquet.
python -m mibici.dataset reproduce --manifest references/data-manifest-2026-10-v2.json --data-dir data/monitoring/source
python -m mibici.monitoring prepare
python -m mibici.monitoring verify

# Entorno exclusivo del reporte (conservar intacto el de API/entrenamiento).
python -m venv .venv-monitoring
# macOS/Linux:
.venv-monitoring/bin/python -m pip install -r requirements-monitoring.txt
.venv-monitoring/bin/python -m pip check
.venv-monitoring/bin/python -m mibici.monitoring report
.venv-monitoring/bin/python -m pytest tests/test_monitoring_evidently.py -q

# Windows PowerShell: sustituir .venv-monitoring/bin/python por
# .venv-monitoring\Scripts\python.exe
```

La preparación utiliza `requirements.txt`; Evidently usa su lock independiente.
Para recuperar el catálogo histórico cuyo enlace dejó de estar disponible,
consultar [CI_CATALOGO.md](CI_CATALOGO.md); conservar la copia original verificada.
El procesamiento no cambia esos hashes ni los manifiestos anteriores.

## Contrato de datos y consumo por el equipo

`data/monitoring/p1/` contiene cuatro Parquet y `metadata.json`, excluidos de Git.
Columnas ordenadas: `station_id, week_start, departures, lag_1, mean_previous_4,
lag_52, week_sin, week_cos`. Clave única estación/semana; objetivo entero
no negativo. Las cinco variables mantienen el orden del modelo/API.
Sólo `lag_52` admite NaN por historia anual insuficiente; HGB los acepta.
El calendario se calcula con la misma fase anual de entrenamiento.

```python
from mibici.monitoring import load_partitions

parts, contract = load_partitions()
train = parts["reference"]
validation = parts["historical_validation"]
evaluation = parts["current"]
features = contract["features"]
X_eval = evaluation[features]
y_eval = evaluation[contract["target"]]
# Usar X_eval/y_eval idénticos y con el mismo orden para A y B.
```

Todos los ejemplos actuales quedan reservados: no entrenar, seleccionar hiperparámetros
ni elegir variables de B con ellos. B puede cambiar hiperparámetros utilizando
reference/validación histórica; registrar después una nueva versión del mismo nombre,
en el mismo Registry de A. Si se desea entrenar con octubre, hace falta acordar otro
corte y regenerar el contrato antes de comparar modelos.

Conservamos MAE como métrica principal y RMSE/WAPE como auxiliares, reutilizando
`mibici.modeling.p1.regression_metrics`. El comparador debe incluir el
`contract_id`, las claves/orden de evaluación y las identidades de ambos modelos.
El segundo integrante implementa entrenamiento/registro/comparación; el tercero
consume los resultados de este reporte y de A/B para interpretación y recomendación.

Los hashes lógicos fijan contenido y orden con fechas ISO, NaN explícitos y nueve
decimales para variables flotantes. Absorben diferencias de serialización de Parquet
y redondeo de plataforma; no redondeamos las variables que recibe el modelo.
Los hashes originales de archivos fuente siguen comprobándose de forma estricta.
No se publican viajes individuales ni tablas de evaluación.

## Reporte de Evidently

Ejecutamos Evidently0.7.23 con `DataDriftPreset` y `DataSummaryPreset`,
ambos sobre las cinco variables numéricas. Fijamos Wasserstein normalizada y
umbral exploratorio0.1 antes del reporte. La distancia se divide entre la desviación
estándar de referencia (con un mínimo de0.001, según Evidently); alerta si distancia≥0.1.
El umbral no constituye una regla de reentrenamiento ni una decisión de promoción.

Artefactos en [reports/monitoring/p1](../reports/monitoring/p1/):

- `evidently.html`: reporte interactivo autónomo; descargar y abrir en navegador.
- `evidently.json`: salida nativa de Evidently.
- `evidently-distributions.png`: captura del HTML nativo con distribuciones y resultados por variable.
- `summary.json`: contrato estable con scores/umbrales/alertas y calidad de ambos conjuntos.
- `drift.csv`: una fila por variable para el módulo de interpretación.
- `drift-quality.png`: figura derivada de los resultados reales de Evidently y faltantes.
- `data-contract.json`: copia pública de los metadatos, sin registros individuales.

El HTML incluye distribuciones agregadas de las cinco variables, sin identificadores
de usuarios, estaciones ni ejemplos de viajes. Ejecutamos localmente con telemetría
desactivada. Los resultados de calidad distinguen duplicados de claves de filas con
variables idénticas: el calendario repetido por estación es esperado.

La ejecución fijada señala cuatro de cinco variables. Faltan3462 valores de
`lag_52` en reference (58.41%) y18 en current (7.69%), sin otros faltantes ni infinitos.
La ausencia anual refleja la historia disponible y se mantiene visible, sin imputarla.
El conjunto actual tiene tres semanas y estaciones repetidas; las variables de calendario
comparan octubre con varios meses. Estas limitaciones deben considerarse antes de
interpretar los cambios o relacionarlos con desempeño.

Las alertas no prueban pérdida de precisión. La interpretación, la tabla A/B y la
recomendación de Continuous Training se completan con los bloques siguientes.

Referencias: [datos abiertos MiBici](https://www.mibici.net/es/datos-abiertos/),
[reportes Evidently](https://docs.evidentlyai.com/docs/library/report),
[Data Summary](https://docs.evidentlyai.com/metrics/preset_data_summary),
[exportación](https://docs.evidentlyai.com/docs/library/output_formats).
