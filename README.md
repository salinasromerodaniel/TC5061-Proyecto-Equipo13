# Proyecto MiBici — Equipo 13

Proyecto integrador de TC5061: demanda semanal P1, riesgo indirecto P2 y agrupación de estaciones P3. Repositorio público: https://github.com/salinasromerodaniel/TC5061-Proyecto-Equipo13.

## Estado de la segunda entrega

La parte A incorpora MLflow Project, entorno fijo, servidor local con SQLite y entradas parametrizadas para las tres preguntas. La parte B de P1 compara un baseline y tres candidatos con nested runs, modelos cargables, firma, ejemplo, manifiesto, configuración y gráficas. La experimentación adicional de P2/P3 y el registro de los tres ganadores en producción se incorporan por sus responsables.

Los datos conservan octubre de 2024–septiembre de 2026: 9,043,167 viajes limpios de toda la red, 758 duplicados removidos y catálogo de 484 estaciones. P1 usa el polígono oficial INEGI de Zapopan y una cohorte histórica de 78 estaciones. P2 aplica el filtro municipal y un proxy de déficit neto, no inventario observado. P3 conserva el alcance del catálogo completo de su baseline anterior; su responsable debe justificar cualquier cambio de alcance antes de comparar candidatos.

## Instalación desde un clon limpio

Python 3.12, declarado como 3.12.15 en `python_env.yaml`. `requirements.txt` fija dependencias directas y transitivas; `python_env.yaml` lo reutiliza y declara las herramientas de instalación, sin mantener otra lista de versiones de ejecución.

macOS/Linux:

```bash
git clone https://github.com/salinasromerodaniel/TC5061-Proyecto-Equipo13.git
cd TC5061-Proyecto-Equipo13
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -e .
```

Windows PowerShell, dentro del clon:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -e .
```

Mantén el entorno activado en cada terminal. Con `--env-manager=local`, MLflow ejecuta el `python` del PATH; no basta con invocar una ruta absoluta al ejecutable de MLflow sin activar el entorno. El punto de entrada comprueba Python 3.12 y todas las versiones del lock antes de procesar datos. Este modo evita depender de pyenv para ejecutar el proyecto en Windows. MLflow también admite crear el entorno declarado con `--env-manager=virtualenv` cuando se dispone de sus herramientas.

## Servidor de tracking y Model Registry

En una terminal con el entorno activado:

```bash
python -m mibici.project server
```

El comando toma host, puerto y SQLite de `config.py`; por defecto abre http://127.0.0.1:5013, usa `mlflow_tracking.db` y conserva el tracking previo. La misma base respalda el Model Registry. No publiques esa base ni sus artefactos.

En una segunda terminal, activa el mismo entorno y configura la conexión:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5013
```

En PowerShell:

```powershell
$env:PYTHONUTF8 = "1"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5013"
```

En Windows configura `PYTHONUTF8=1` también al redirigir logs: MLflow imprime símbolos Unicode y una consola o salida capturada en CP1252 puede rechazarlos.

Cada computadora usa su servidor local y sus artefactos. Los nombres coincidentes de experimento no sincronizan bases ni modelos entre computadoras. Para la integración final, el responsable del registry debe ejecutar o disponer de las corridas y artefactos completos en el servidor donde registrará los ganadores.

## Entradas de MLflow Project

`MLproject` ofrece `ingest` y `train`. Los parámetros usan el marcador `config`: Python lo resuelve contra `mibici/config.py`, evitando duplicar rutas, fechas y semillas en YAML. Los parámetros explícitos sustituyen esos defaults. Todos los comandos nuevos se lanzan con `mlflow run`, desde la raíz del clon y con código/dependencias guardados en un commit.

### Ingesta y verificación

Reconstruir los 24 meses del manifiesto v2, descargando originales ausentes y comprobando los disponibles:

```bash
mlflow run . --env-manager=local --experiment-name mibici-equipo13 --run-name DATA_reproduce -e ingest
```

Verificar el conjunto existente sin descargar ni transformar:

```bash
mlflow run . --env-manager=local --experiment-name mibici-equipo13 --run-name DATA_verify -e ingest -P mode=verify
```

Si conservas los viajes anteriores pero el catálogo todavía usa la conversión inicial:

```bash
mlflow run . --env-manager=local --experiment-name mibici-equipo13 --run-name DATA_repair -e ingest -P mode=repair-stations
```

La reparación exige todos los hashes de fuentes y viajes y reemplaza solo el catálogo, tras comprobar su hash v2. Opciones `manifest` y `data_dir` permiten rutas personalizadas. Los manifiestos antiguos se conservan para trazabilidad; verificar un catálogo v2 con el manifiesto v1 falla correctamente. Reglas y condiciones de uso en [DATOS.md](references/DATOS.md).

### P1: búsqueda y reproducción del elegido

Comparar baseline y tres candidatos Poisson con 7, 15 y 31 hojas máximas:

```bash
mlflow run . --env-manager=local --experiment-name mibici-equipo13 --run-name P1_search -e train -P question=p1 -P model=hgb-search
```

Reproducir el candidato elegido por validación, junto con el baseline de referencia:

```bash
mlflow run . --env-manager=local --experiment-name mibici-equipo13 --run-name P1_selected -e train -P question=p1 -P model=hgb -P params_file=configs/p1/selected.json -P report_dir=reports/delivery2/p1-selected
```

El JSON contiene overrides de hiperparámetros del estimador. La semilla común vive en `config.py`. El padre agrupa hijos baseline/candidatos. Se elige por MAE de validación antes de registrar las métricas de prueba. Cada ejecución crea sus propios identificadores y mantiene el historial de MLflow; los archivos del directorio de resultados indicado se actualizan.

P1 seleccionado: HGB Poisson con 7 hojas, MAE de validación 14.4154 y MAE de prueba 16.9977, frente a 15.9787 y 17.3734 del baseline. Mejora de MAE de prueba: 2.16 %. RMSE de prueba empeora respecto al baseline; no se afirma superioridad en todas las métricas. Comparación, límites y artefactos en [ENTREGA2_P1.md](references/ENTREGA2_P1.md).

### Contrato de P2/P3

Los responsables pueden ejecutar los modelos actuales con el mismo entorno, servidor y manifiesto:

```bash
mlflow run . --env-manager=local --experiment-name mibici-equipo13 --run-name P2_baseline -e train -P question=p2 -P model=baseline
mlflow run . --env-manager=local --experiment-name mibici-equipo13 --run-name P2_rf -e train -P question=p2 -P model=random_forest
mlflow run . --env-manager=local --experiment-name mibici-equipo13 --run-name P3_kmeans -e train -P question=p3 -P model=kmeans
```

`params_file` admite un JSON de overrides, por ejemplo `n_estimators`/`max_depth` para Random Forest o `n_clusters` para KMeans. Los valores base están en `config.py`. La integración corrige el parámetro de grupos registrado por P3 y permite que respete el servidor indicado; no cambia su representación ni sus métricas históricas. Los ganadores definitivos y comandos seleccionados de P2/P3 se añadirán cuando termine su experimentación.

## Entrega al responsable de Model Registry

`reports/delivery2/p1/registry_handoff.json` identifica modelo elegido, URI cargable, run, padre, commit, versión de datos y nombre sugerido `mibici-equipo13-p1-demanda`. Este bloque todavía no registra una versión en producción ni asigna alias: corresponde a la parte C. El esquema de ciclo de vida elegido por el equipo y los comandos de carga por nombre deben incorporarse aquí al integrar C.

## Validación

```bash
python -m pytest -q -W error::DeprecationWarning
```

Las integraciones opcionales se activan con `MIBICI_CATALOG_INTEGRATION=1` y `MIBICI_PROJECT_INTEGRATION=1`. La primera verifica el catálogo oficial; la segunda ejecuta una entrada real de MLflow Project contra HTTP con datos sintéticos. GitHub Actions las ejecuta en Windows, Linux y macOS. Alcance y evidencias de clon limpio en [ENTREGA2_A.md](references/ENTREGA2_A.md).

## Estructura y colaboración

- `MLproject`, `python_env.yaml`: paquete ejecutable y entorno.
- `mibici/project.py`, `tracking.py`, `modeling/train.py`: CLI, tracking y despacho.
- `mibici/modeling/p1_candidates.py`: candidatos P1; `p1.py` conserva baselines históricos.
- `references/`: manifiestos y documentación técnica; `reports/delivery2/p1/`: resultados agregados.
- `data/`, `models/`, `mlruns/`, `mlartifacts/` y bases de datos: solo locales y excluidos de Git.

Seguimos GitHub Flow con main protegida: una rama por cambio, PR revisado por otro integrante y sin saltarse la protección. En esta entrega cada integrante abre al menos un PR y revisa uno ajeno. Convenciones: `feat/`, `fix/`, `docs/` y commits Conventional Commits.

Equipo: Daniel Salinas Romero, José Alberto Velázquez Trujillo y Diego Armando Vazquez Calderon. Daniel desarrolla A y B/P1; los otros bloques se integran con sus responsables.

Plantilla oficial Cookiecutter Data Science; procedencia en `references/scaffold-config.json`. Datos, modelos y tracking no se suben al repositorio. La documentación de la primera entrega permanece como referencia histórica.

## Parte B — P2: comparación de candidatas

Con el entorno activo y `MLFLOW_TRACKING_URI` apuntando al servidor del equipo:

```bash
mlflow run . --env-manager=local --experiment-name mibici-equipo13 --run-name P2_search -e train -P question=p2 -P model=search
mlflow run . --env-manager=local --experiment-name mibici-equipo13 --run-name P2_selected -e train -P question=p2 -P model=logistic -P params_file=configs/p2/selected.json -P report_dir=reports/delivery2/p2-selected
```

La búsqueda P2 registra el Dummy original y tres candidatas: RF de 200 árboles y
profundidad 12, RF de 300 árboles y profundidad 6, y regresión logística balanceada
con escalado. Las combinaciones se definen en `P2_SEARCH_CONFIGS` de `config.py`.
La regresión logística se selecciona por F1 de validación: 0.607450 frente a
0.402189 del baseline; su F1 de prueba es 0.638596. La selección no usa prueba.

Se conservan la cohorte de 73 estaciones, los cortes temporales y la etiqueta
indirecta de déficit de bicicletas. No se observan inventarios ni stockouts reales.
Cada hija registra modelo con firma y ejemplo, manifiestos, configuración,
matrices y curvas precisión-recall, además de métricas y tiempos. Se verifica la
carga del modelo y la igualdad de predicciones. Las carpetas `plots/`, `data/`,
`config/` y `evaluation/` organizan los artefactos; no se registran usuarios.

Resultados: `reports/delivery2/p2/comparison.csv`, `results.json`,
`split_metadata.json` y `registry_handoff.json`. Este último entrega a C el modelo
cargable y su linaje; no implica que ya esté registrado ni promovido a producción.
Una reproducción crea runIDs y tiempos nuevos. Los detalles y límites de P2 se
encuentran en [references/P2.md](references/P2.md).
