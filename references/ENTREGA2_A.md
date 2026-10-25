# Parte A — Proyecto reproducible

Autor: Daniel Salinas Romero. Equipo 13. Fecha: 25 de octubre de 2026.

## Empaquetado y configuración

El repositorio conserva la ingesta y los manifiestos de datos de la primera entrega y añade `MLproject`, con entradas ingest y train, y `python_env.yaml`, con Python 3.12.15, herramientas de instalación fijadas y las dependencias de requirements.txt. El paquete editable apunta al código del clon. MLflow Projects permite declarar conjuntamente las entradas y el entorno para ejecutar desde código, sin depender de notebooks (MLflow contributors, s. f.-a).

Los defaults YAML usan el marcador config y se resuelven en mibici/config.py. Así no se repiten rutas, fechas ni semillas en el archivo MLproject. Las opciones explícitas permiten otro manifiesto, directorio de datos/resultados, pregunta, modelo y un JSON de hiperparámetros. La semilla común se cambia en config.py para preservar su trazabilidad.

El dispatcher de train invoca los runners P1/P2/P3. P1 admite baseline, hgb y hgb-search; P2 admite baseline/Dummy y Random Forest; P3 admite KMeans. Los hiperparámetros de los modelos existentes pueden variarse sin editar MLproject. Los responsables de P2/P3 incorporarán otros algoritmos si su experimentación lo necesita.

## Tracking local con base de datos

```bash
python -m mibici.project server
```

Este comando resuelve SQLite, host y puerto desde config.py y levanta MLflow en http://127.0.0.1:5013. La base mlflow_tracking.db conserva el tracking previo y respalda también Model Registry. Se usa un servidor local sencillo para el alcance académico; no se publica como servicio de producción. El registro de modelos requiere almacenamiento respaldado por base de datos (MLflow contributors, s. f.-b).

Con el entorno activado en la terminal de ejecución:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5013
```

En PowerShell se configuran `$env:PYTHONUTF8 = "1"` y `$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5013"`. UTF-8 evita errores al capturar los símbolos Unicode de la salida de MLflow. Los puntos de entrada retoman el run creado por MLflow Projects antes de abrir hijos. P1 organiza su búsqueda en hijos; las pruebas de integración de P2/P3 registran sus baselines como hijos del run de Project. La corrección de integración de P3 registra su n_clusters real y respeta el servidor indicado.

Las computadoras no sincronizan automáticamente SQLite ni artefactos por usar el mismo nombre de experimento. Para la entrega conjunta, los responsables deben reproducir o reunir los runs completos en el servidor donde C registrará los tres ganadores. No se suben las bases ni modelos a Git.

## Comandos reproducibles

El README contiene instalación desde clon limpio, activación del entorno y configuración del servidor para Windows y macOS/Linux. Se usa --env-manager=local con el entorno fijado ya instalado; la entrada comprueba versiones antes de ejecutar. El entorno también queda declarado para el gestor virtualenv de MLflow.

```bash
mlflow run . --env-manager=local --experiment-name mibici-equipo13 --run-name DATA_reproduce -e ingest
mlflow run . --env-manager=local --experiment-name mibici-equipo13 --run-name DATA_verify -e ingest -P mode=verify
mlflow run . --env-manager=local --experiment-name mibici-equipo13 --run-name P1_selected -e train -P question=p1 -P model=hgb -P params_file=configs/p1/selected.json -P report_dir=reports/delivery2/p1-selected
mlflow run . --env-manager=local --experiment-name mibici-equipo13 --run-name P2_baseline -e train -P question=p2 -P model=baseline
mlflow run . --env-manager=local --experiment-name mibici-equipo13 --run-name P3_kmeans -e train -P question=p3 -P model=kmeans
```

Ingest usa por defecto la reconstrucción de los 24 meses declarados, reutilizando únicamente originales con hash correcto. mode=verify comprueba el conjunto local y mode=repair-stations reconstruye solo el catálogo v2 cuando los demás archivos ya coinciden. No se acepta una versión de fuente modificada silenciosamente. Los comandos seleccionados definitivos de P2/P3 deberán sustituir los ejemplos de baseline cuando sus responsables concluyan B.

## Evidencia de ejecución

Implementación original comprobada en commit `33f9d899fd27e173624c615a808b9c629a627cfe`.

| Entrada | Run padre | Resultado |
|---|---|---|
| ingest, mode=verify | 86295ee87c864e369dce779af38f1abe | Verificó los 24 meses y registró el manifiesto. |
| train, p1/hgb-search | 8f09f3642a404fbabff4572ec245b5ea | Baseline y tres candidatos, firmas, modelos recargados y artefactos. |
| train, p2/baseline | 12d05a8ded0a4388bbdb348b2ccd5aa0 | Runner municipal y tracking común comprobados con Dummy. |
| train, p3/kmeans | 4fd8292cecff4463829c58336ab3a4df | Runner y tracking común comprobados con cuatro grupos. |

P2/P3 se ejecutaron para validar el contrato de A; no constituyen sus tres candidatos adicionales de B ni una selección nueva de ganadores.

## Prueba de clon limpio y límites

Se creó un clon limpio del commit indicado, un entorno Python 3.12 nuevo y una base de tracking independiente. Se instalaron requirements.txt y el paquete editable, pip check pasó, y se ejecutaron ingest y train desde MLflow Projects contra HTTP. Solo se reutilizaron los CSV y GeoJSON originales como caché verificada para evitar otra transferencia; todos los Parquet se reconstruyeron. No se reutilizaron modelos, Parquet ni la base de tracking.

La reconstrucción mantuvo los hashes del manifiesto v2. Pasaron las 68 pruebas de ese commit, con la prueba de red opcional omitida; los tres candidatos, el baseline, sus particiones y el modelo seleccionado reprodujeron exactamente las métricas originales. Los nuevos runIDs y tiempos son propios de la nueva ejecución.

Se añadieron después casos de error cero del baseline y una prueba del CLI real contra HTTP con datos sintéticos para la matriz Windows/Linux/macOS. La matriz pasó en Windows, Linux y macOS en [GitHub Actions 40227832428](https://github.com/salinasromerodaniel/TC5061-Proyecto-Equipo13/actions/runs/40227832428). Se corrigió un fallo de codificación CP1252 al capturar la salida Unicode de MLflow en Windows mediante PYTHONUTF8=1 y decodificación explícita UTF-8. No se confunde la prueba sintética del CLI con una reconstrucción de 9 millones de viajes en cada sistema. La prueba completa descrita se ejecutó en macOS ARM64.

Dos intentos iniciales lanzaron MLflow sin activar el entorno y fallaron porque el subproceso no encontraba python. Se conservaron como fallos de configuración identificados; el README exige la activación. No son resultados de entrenamiento.

## Colaboración y exclusiones

Código y documentación entran a main por PR revisado. Cada integrante abre y revisa al menos un PR en esta entrega. .gitignore excluye datos, bases, carpetas de tracking y modelos serializados. Las salidas locales de las pruebas de contrato P2/P3 quedan fuera del commit; sus evidencias se conservan en MLflow y en este resumen.

## Referencias

MLflow contributors. (s. f.-a). *MLflow Projects*. Recuperado el 25 de octubre de 2026, de https://mlflow.org/docs/latest/ml/projects/

MLflow contributors. (s. f.-b). *Model Registry workflows*. Recuperado el 25 de octubre de 2026, de https://mlflow.org/docs/latest/ml/model-registry/workflow/

MLflow contributors. (s. f.-c). *MLflow Tracking APIs*. Recuperado el 25 de octubre de 2026, de https://mlflow.org/docs/latest/ml/tracking/tracking-api/
