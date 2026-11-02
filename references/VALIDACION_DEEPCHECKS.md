# Ejecución de Deepchecks: P1, P2 y P3

Validación realizada en Windows con Python 3.12. La carpeta de ejecución es `reports/deepchecks/real_20261102T050329713596Z`; el sello está en UTC (la ejecución corresponde al 1 de noviembre de 2026 en Ciudad de México).

Base del repositorio: commit `2d4f05b`, con la contribución local de Deepchecks aún sin commit. Las huellas del código de validación y el entorno completo se guardan en `execution-evidence.json` y `environment-freeze.txt` dentro de la carpeta local de resultados. No se afirma que estas modificaciones locales pertenezcan al commit base.

## Comandos ejecutados

```powershell
.\.venv-deepchecks\Scripts\python.exe -m pip check
.\.venv-deepchecks\Scripts\python.exe -m unittest discover -s tests -p test_validation.py -v
.\.venv-deepchecks\Scripts\python.exe -m mibici.validation_tasks export
.\.venv-deepchecks\Scripts\python.exe -m mibici.validation_tasks check
```

`pip check`: **No broken requirements found.** Pruebas de esta contribución: **19 aprobadas, cero fallidas y cero omitidas**. No se presenta este resultado como ejecución de toda la suite del equipo.

Las tres exportaciones verificaron los archivos de la ingesta y el manifiesto. Versión de datos: `6622c0fec77e6e66f2bdb91f5761af0a792c7d3ce19591d35afc2f842a2ff9a3` (24 meses, octubre de 2024 a septiembre de 2026). P1/P2 usaron el polígono INEGI fijado con SHA-256 `b7b9207074bc0dbe807a43eceb6f02dafb6a8c1ede27484c05fec073e0d6fabe`.

## Resultados reales

| Pregunta | Filas | Integridad | Drift | Estado |
|---|---:|---|---|---|
| P1 | 7,160 estación-semana | Aprobada en train, validation y test | Alertas de calendario en ambas comparaciones | DRIFT_REQUIRES_REVIEW |
| P2 | 7,095 estación-semana, 73 estaciones | Aprobada en train, validation y test | Alertas de flujo neto y etiqueta en ambas comparaciones | DRIFT_REQUIRES_REVIEW |
| P3 | 484 estaciones del catálogo completo | Aprobada en la matriz completa | No aplica: no hay partición train/test en el pipeline actual | PASS |

**Ningún chequeo quedó sin ejecutar.** Se generaron 11 HTML (cinco P1, cinco P2, uno P3), sus JSON y los resúmenes. El comando final devolvió código 0 porque el drift se configuró como alerta revisable; no significa que el drift haya aprobado.

P1 informa 3,559 nulos estructurales de `lag_52`. No se imputaron para superar los chequeos: el candidato original acepta esa ausencia de historia estacional y el baseline estacional usa fallback. Las demás columnas de P1 no presentaron nulos. La integridad de P3 corresponde a la salida de `p3_station_features`, cuya implementación ya rellena nulos con cero; no acredita ausencia de faltantes antes de esa transformación.

## Interpretación de drift

- **P1:** KS en `week_sin`/`week_cos`: aproximadamente 0.46/0.92 en validación y 0.75/0.64 en prueba, por encima del umbral exploratorio 0.2. Las etiquetas continuas permanecen dentro del umbral (aproximadamente 0.14 y 0.03). El cambio en calendario es coherente con comparar un periodo largo de entrenamiento con bloques de ocho semanas; por sí solo no demuestra deterioro del modelo. Conviene interpretar las métricas por periodo/estación sin modificar el test.
- **P2:** KS en `net_flow`/`net_flow_mean4`: aproximadamente 0.24/0.21 en validación y 0.26/0.33 en prueba. Drift de etiqueta, Cramer's V, aproximadamente 0.27 en ambos bloques. La proporción positiva baja de 3,083/5,927 (52.02%) en entrenamiento a 147/584 (25.17%) en validación y 148/584 (25.34%) en prueba. Se debe revisar la estabilidad temporal del proxy y el balance entre precisión/recall; no alterar el test ni confundir el proxy con stockouts observados.
- **P3:** no se inventa una partición ni etiquetas para conseguir un reporte de drift. Una comparación de ventanas temporales sería una extensión del pipeline de clustering, no parte de su protocolo actual.

El umbral 0.2 es diagnóstico y no universal; KS y Cramer's V no son medidas intercambiables. Los avisos de deprecación de las dependencias no impidieron ejecutar las suites.

## Capturas pendientes

La apertura automatizada de los HTML locales fue rechazada por la política de URL del navegador (`file://` no permitido). Los archivos existen y los resultados se verificaron por sus JSON; no se fabricaron capturas. Para cerrar la evidencia visual:

1. Ejecutar el comando de pruebas y capturar el resumen `Ran 19 tests ... OK`.
2. Ejecutar `mibici.validation_tasks check` y capturar los tres estados finales.
3. Abrir con doble clic los HTML de integridad y drift, y capturar las condiciones de P1/P2 y la integridad de P3.

Los reportes completos, el entorno y los datos permanecen fuera de Git. El PR debe contener código, requisitos, instrucciones y esta síntesis; añadir las capturas revisadas cuando estén disponibles.

La síntesis verificable y la salida de pruebas se versionan en `reports/figures/evidence/delivery3/deepchecks-execution.json` y `deepchecks-tests.txt`. Los HTML completos permanecen locales por su tamaño.
