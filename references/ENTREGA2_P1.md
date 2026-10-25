# Parte B — P1: experimentación y artefactos

Autor: Daniel Salinas Romero. Equipo 13. Fecha: 25 de octubre de 2026.

## Pregunta, datos y comparación

Pronosticar viajes iniciados por estación de Zapopan durante la siguiente semana. La unidad es estación-semana, con origen el lunes a las 00:00 de la hora publicada por MiBici. Se conservan los 24 meses de octubre de 2024 a septiembre de 2026, el manifiesto de transformación 2 y la delimitación INEGI de la primera entrega. No se estima demanda no atendida, inventario disponible ni estaciones nuevas.

Versión de datos: `6622c0fec77e6e66f2bdb91f5761af0a792c7d3ce19591d35afc2f842a2ff9a3`. Las 106 estaciones geográficas de referencia producen una cohorte histórica de 78 estaciones. Se conserva exactamente el panel de la primera entrega: 5,927 filas de entrenamiento, 609 de validación y 624 de prueba. Validación `[2026-06-08, 2026-08-03)` y prueba `[2026-08-03, 2026-09-28)`.

El baseline de persistencia usa los viajes de la semana anterior. Su nueva ejecución coincide exactamente con las métricas del run histórico v2, sobre los mismos ejemplos. El protocolo es de origen móvil semanal: una semana ya finalizada puede aportar observaciones a la siguiente; no es un pronóstico de ocho semanas emitido en una sola fecha.

## Variables y candidatos

Los candidatos utilizan lag de una semana, media de las cuatro semanas anteriores, lag de 52 semanas y seno/coseno del calendario de la semana objetivo. El calendario es conocido al pronosticar. No se usa el objetivo de esa semana, nombres de estaciones, IDs numéricos como predictores ni campos personales. La historia anual ausente se conserva como NaN y la gestiona el estimador.

Se ejecutaron tres configuraciones de `HistGradientBoostingRegressor(loss="poisson")`, con 7, 15 y 31 hojas máximas. La pérdida de Poisson se adapta a conteos no negativos; el método maneja valores faltantes (scikit-learn developers, s. f.-a). Parámetros comunes: 150 iteraciones, learning_rate=0.05, min_samples_leaf=20, l2_regularization=1.0, semilla 13 y early_stopping=False. Se desactiva la validación aleatoria interna para mantener el corte temporal. Los ejemplos de retardos de scikit-learn ilustran por qué debe evitarse una separación aleatoria en pronósticos temporales (scikit-learn developers, s. f.-b).

La búsqueda tiene un padre y cuatro hijos: baseline y tres candidatos. Se ajusta cada candidato solo con train. Se congela el elegido por MAE de validación antes de registrar las métricas de prueba. La prueba ya formó parte del desarrollo de la entrega anterior; se conserva por comparabilidad y no se presenta como un holdout externo nuevo.

## Resultados

| Modelo | Hojas máximas | MAE validación | MAE prueba | RMSE prueba | Mejora MAE prueba |
|---|---:|---:|---:|---:|---:|
| previous_week | — | 15.9787 | 17.3734 | 27.9490 | 0.00% |
| hgb_1_leaves7 | 7 | 14.4154 | 16.9977 | 28.3762 | 2.16% |
| hgb_2_leaves15 | 15 | 14.6743 | 16.5617 | 27.0019 | 4.67% |
| hgb_3_leaves31 | 31 | 15.0796 | 16.8845 | 27.6760 | 2.81% |

La mejora es `100 × (MAE_baseline − MAE_modelo) / MAE_baseline`, calculada por conjunto. El porcentaje no se define si el baseline tiene error cero. MAE y RMSE están en viajes por estación-semana. WAPE y todas las métricas de train/validación/prueba están en el archivo de resultados.

## Selección, estabilidad y costo

Se propone el modelo de 7 hojas por su menor MAE de validación: mejora 9.78% frente a persistencia. En prueba, su MAE mejora 2.16%. El candidato de 15 hojas obtuvo mejor MAE de prueba, pero no sustituye al elegido: hacerlo usaría la prueba para seleccionar.

El modelo elegido no domina todas las métricas. Su RMSE de prueba es 28.3762 frente a 27.9490 de persistencia. El MAE por semana de prueba varía entre 11.3738 y 30.3010; en el baseline, entre 11.7308 y 26.8718. La mejora agregada convive con una semana peor y con errores grandes más costosos. Esto exige monitoreo por semana y estación antes de interpretar el modelo como una mejora operativa uniforme.

En esta máquina el ajuste tardó aproximadamente 0.36 s para 7 hojas, 0.87 s para 15 y 1.44 s para 31. Cargar y predecir el lote de prueba tomó aproximadamente 0.078 s para el elegido. Esta última medida incluye carga del modelo y no es un benchmark de latencia por solicitud. La menor complejidad y costo de ajuste refuerzan la elección por validación, sin eliminar los límites de estabilidad.

## Tracking y artefactos

Experimento `mibici-equipo13`. Commit ejecutado: `33f9d899fd27e173624c615a808b9c629a627cfe`. Padre de búsqueda: `8f09f3642a404fbabff4572ec245b5ea`. Los hijos incluyen etiquetas question, model_family, stage_of_work, team y git_commit.

Cada hijo guarda configuración, MLproject, entorno y lock; manifiesto de datos y geografía; partición; modelo con firma y ejemplo; dataset de entrenamiento con digest/esquema mediante log_input; gráficas real contra pronosticado y residuales de prueba; y errores por semana. log_input registra metadatos del DataFrame, sin subir filas de viajes. Los ejemplos contienen únicamente variables agregadas de demanda.

La firma declara cinco variables double, con lag_52 opcional, y una salida numérica por fila. La persistencia se empaqueta mediante Models From Code; los HGB se guardan con el formato skops y el tipo interno TreePredictor explícitamente autorizado para los estimadores entrenados. Se recargaron los cuatro modelos y se comprobó igualdad de sus predicciones con las calculadas en memoria.

Resultados versionados: `reports/delivery2/p1/results.json`, `comparison.csv` y `registry_handoff.json`. Los manifiestos y resultados anteriores se conservan como historia. Una reproducción crea otros runIDs y tiempos, manteniendo el protocolo y las métricas numéricas comprobadas.

## Reproducción y entrega a C

Con el entorno activo, tracking HTTP configurado y datos reconstruidos mediante la parte A:

```bash
mlflow run . --env-manager=local --experiment-name mibici-equipo13 --run-name P1_search -e train -P question=p1 -P model=hgb-search
mlflow run . --env-manager=local --experiment-name mibici-equipo13 --run-name P1_selected -e train -P question=p1 -P model=hgb -P params_file=configs/p1/selected.json -P report_dir=reports/delivery2/p1-selected
```

Modelo elegido: run `65ed8a9510984cf2aedde1663050440a`, URI `models:/m-0014a816ca2547389d4e203d11dc5537`. El archivo registry_handoff.json conserva linaje y sugiere el nombre `mibici-equipo13-p1-demanda`. El responsable de C debe disponer del run y sus artefactos en su servidor, registrar la versión, describirla y verificar la carga por nombre/alias. Esta sección entrega un modelo cargable, pero no afirma que ya esté registrado en producción.

## Referencias

MLflow contributors. (s. f.-a). *MLflow Projects*. Recuperado el 25 de octubre de 2026, de https://mlflow.org/docs/latest/ml/projects/

MLflow contributors. (s. f.-b). *MLflow Tracking APIs*. Recuperado el 25 de octubre de 2026, de https://mlflow.org/docs/latest/ml/tracking/tracking-api/

MLflow contributors. (s. f.-c). *MLflow Models*. Recuperado el 25 de octubre de 2026, de https://mlflow.org/docs/latest/ml/model/

scikit-learn developers. (s. f.-a). *HistGradientBoostingRegressor*. Recuperado el 25 de octubre de 2026, de https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html

scikit-learn developers. (s. f.-b). *Lagged features for time series forecasting*. Recuperado el 25 de octubre de 2026, de https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html

## Evidencia visual

Capturas reales de MLflow en reports/figures/evidence/delivery2/: 01-p1-nested-runs.png (padre y cuatro hijos), 02-p1-comparison.png (MAE de validación), 03-p1-run-detail.png (parámetros y métricas), 04-p1-artifacts.png (artefactos) y 05-p1-model-signature.png (firma y ejemplo del modelo). Las gráficas exportables de los cuatro hijos están en reports/delivery2/p1/plots/. El documento compartido incorpora seis figuras de esta evidencia.
