# Proyecto MiBici — Equipo 13

Primera entrega del proyecto integrador de TC5061: propuesta y base reproducible para estudiar la demanda y los patrones de uso de MiBici en el Área Metropolitana de Guadalajara.

## Preguntas

| ID | Pregunta | Tarea |
|---|---|---|
| P1 | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicicletas o anclajes? | Clasificación |
| P3 | ¿Qué estaciones comparten patrones de uso y dónde se ubican? | Clustering |

La formulación inicial propone estaciones de Zapopan y horizonte semanal para P1/P2. El equipo debe validar ese alcance con los datos. El riesgo de P2 requiere una aproximación justificada: los viajes históricos no proporcionan inventario de bicicletas por estación.

## Estado

Ingesta automática, normalización, Parquet y manifiestos implementados. Periodo de P1: octubre de 2024–septiembre de 2026, con **9,043,167 viajes limpios de toda la red** (758 duplicados removidos). El filtro geográfico de Zapopan usa el polígono oficial INEGI y el catálogo de septiembre de 2026; 826,486 inicios quedan asociados a estaciones dentro del polígono de referencia.

P1 compara persistencia, media de cuatro semanas y un baseline estacional de 52 semanas. La prueba usa las mismas 78 estaciones de la cohorte histórica y los mismos ejemplos para los tres modelos. Persistencia fue elegida por validación: MAE de prueba **17.37 viajes por estación-semana**. Datos, partición, límites y resultados en [references/P1.md](references/P1.md). P2, P3, su integración común y el notebook exploratorio siguen a cargo de sus responsables.

## Estructura

```text
mibici/
  config.py             # Configuración y rutas
  dataset.py            # Ingesta, verificación y reproducción
  features.py           # Agregaciones y variables históricas de P1
  modeling/
    train.py            # Pendiente: orquestación común P1/P2/P3
    predict.py          # Plantilla
    p1.py               # Baselines P1, evaluación y MLflow
  plots.py              # Diagrama de arquitectura
data/                   # Datos locales, excluidos de Git
models/                 # Modelos locales, excluidos de Git
notebooks/              # Exploración
references/             # Manifiesto; pendientes ML Canvas y arquitectura
reports/figures/         # Figuras
```

`data/` está excluido de Git y no aparece en el clon inicial; la ingesta crea sus directorios automáticamente. Los notebooks usarán nombres como `1.0-dsr-exploracion.ipynb`. La lógica final debe estar en el módulo `mibici`.

## Instalación

Python **3.12** (validado con 3.12.15). Instalación, 37 pruebas y reproducción de P1 verificadas en un clon y entorno nuevos en macOS ARM64; detalle y alcance de la validación en [references/VALIDACION_P1.md](references/VALIDACION_P1.md). Desde un clon del repositorio y la rama de este cambio:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -e .
```

En Windows puede usarse `py -3.12 -m venv .venv` y activar con `.venv\Scripts\Activate.ps1` en PowerShell. La instalación Windows no se ha validado. No hace falta uv para ejecutar el proyecto.

`requirements.txt` fija dependencias directas y transitivas del bloque de ingesta y pruebas. `requirements.in` declara los requisitos de entrada. Incluye las dependencias de P1, visualización y tracking; los paquetes específicos de notebooks y de los demás modelos se incorporarán por sus responsables. Para actualizar el archivo de versiones, desde un entorno con uv 0.13.1:

```bash
uv pip compile --universal --python-version 3.12 --no-emit-index-url requirements.in --output-file requirements.txt
```

## Reconstruir los datos verificados

Este es el comando recomendado al clonar: descarga las URLs exactas del manifiesto, verifica sus SHA-256 y reconstruye los Parquet con las mismas versiones y tamaño de lote.

```bash
python -m mibici.dataset reproduce --manifest references/data-manifest-2026-09.json
```

Los datos se guardan exclusivamente en `data/`. Si ya existen los CSV, se verifica su hash y se reutilizan. Para verificar archivos y conteos sin red:

```bash
python -m mibici.dataset verify --manifest references/data-manifest-2026-09.json
```

Si falta un archivo o cambia su hash, la verificación falla. Una fuente modificada no se acepta silenciosamente: se requiere generar y revisar una versión nueva. La reconstrucción exige igualdad del manifiesto completo, incluidos hashes de Parquet. La versión de datos está en `data_version` del JSON y podrá registrarse en MLflow.

## Ingestar un periodo nuevo

Para generar una versión nueva, usar una ruta de manifiesto que todavía no exista:

```bash
python -m mibici.dataset ingest --start-month 2026-09 --end-month 2026-09 --manifest references/data-manifest-nueva-prueba.json
```

Se puede ampliar el rango con los argumentos de meses inclusivos. La ingesta descubre los enlaces publicados por MiBici; no supone que todos los archivos tengan el mismo directorio web. El catálogo predeterminado es `nomenclatura_2026_09.csv`; otro catálogo debe elegirse explícitamente con `--stations-file` y estar publicado. La selección del catálogo debe justificarse para el periodo elegido.

No se sobrescribe un manifiesto existente. El procesamiento por lotes admite múltiples meses y mantiene en memoria el índice de deduplicación de un mes a la vez. No ejecutar dos ingestas simultáneas contra el mismo directorio. `--data-dir` permite aislar reconstrucciones. Los cambios de lote o versión de pandas/pyarrow pueden cambiar bytes y, por tanto, la versión de datos.

Se conservan tiempos de origen sin zona horaria: falta confirmar su semántica antes de construir variables temporales. Se descartan campos de usuario de la salida y se conservan estaciones históricas. Las reglas, fuentes y limitaciones están en [references/DATOS.md](references/DATOS.md).

## Pruebas

```bash
python -m pytest -q
```

Los casos del repositorio usan registros sintéticos. La prueba real descarga y reconstruye los datos localmente; no se incluyen CSV, Parquet, modelos ni registros personales en Git.

## Colaboración

Seguimos GitHub Flow: crear una rama desde `main`, trabajar en ella, abrir un Pull Request y solicitar revisión de otro integrante antes de integrar. Tras el primer commit de la plantilla, los cambios deben pasar por PR revisados. La protección de `main` está activada: exige un PR con al menos una aprobación, descarta aprobaciones tras cambios y aplica también al administrador. Se bloquean force-push y eliminación de la rama.

- Ramas: `feat/descripcion`, `fix/descripcion`, `docs/descripcion`, `chore/descripcion`.
- Commits: Conventional Commits, por ejemplo `feat: add monthly trip ingestion`.
- Cada integrante debe aportar commits, abrir al menos un PR y revisar al menos uno de otro compañero.
- No subir datos, carpetas de MLflow, modelos ni credenciales. El manifiesto JSON contiene solo metadatos y hashes, y sí se versiona. Mantener configuración, fechas y semillas en `mibici/config.py`.

## Buenas prácticas previstas de MLflow

Experimento `mibici-equipo13`; al menos un run descriptivo por P1/P2/P3 ejecutado desde código. Registrar rango de fechas, versión de datos, semilla, modelo e hiperparámetros, métricas con conjunto evaluado, artefactos y etiquetas de equipo/pregunta/commit. No registrar datos personales ni credenciales. P1 ya utiliza este contrato; falta integrar P2/P3 en un comando común.

## Equipo y acceso

- Daniel Salinas Romero: datos, reproducibilidad, arquitectura y P1.
- José Alberto Velázquez Trujillo: integrante identificado en el documento colaborativo; rol por confirmar.
- Integrantes restantes y roles: pendientes de confirmación.

El repositorio es público: compañeros y profesor pueden verlo y clonarlo sin invitación. Para subir ramas al mismo repositorio, falta invitar a los compañeros como colaboradores con sus usuarios de GitHub.

## Procedencia

Plantilla oficial: [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/). Opciones de generación en `references/scaffold-config.json`.

## Ejecutar P1 con el periodo definitivo

Tras instalar el entorno y situarse en un commit del código:

```bash
python -m mibici.dataset reproduce --manifest references/data-manifest-2024-10_2026-09.json
python -m mibici.geography
python -m mibici.modeling.p1
python -m mibici.plots
```

El primer comando descarga/reconstruye los 24 meses con sus hashes. El segundo verifica y descarga el polígono INEGI. El tercero registra tres runs P1 en `mibici-equipo13`, usando SQLite local y artefactos en `mlartifacts/`. Exige que código y dependencias estén en un commit para registrar su versión exacta; no depende de un notebook. `run_p1()` puede ser invocado desde el futuro orquestador común.

Métricas y runIDs: `reports/p1/results.json` y `comparison.csv`. Figuras: `reports/figures/p1/`. La cohorte y los cortes de validación/prueba se mantienen comparables con la ejecución anual. La prueba abarca [2026-08-03, 2026-09-28), con origen móvil semanal; el resultado no acredita estaciones nuevas ni demanda no atendida. Las proyecciones anuales no se evalúan.

Para consultar MLflow desde la raíz del repositorio:

```bash
mlflow server --backend-store-uri sqlite:///mlflow_tracking.db --host 127.0.0.1 --port 5013
```

Abrir http://127.0.0.1:5013 y seleccionar el experimento. Se puede usar `--tracking-uri` o `MLFLOW_TRACKING_URI` al ejecutar P1 para apuntar al servidor compartido que el equipo defina. SQLite y las carpetas de artefactos están excluidos de Git. Una nueva ejecución crea nuevos runIDs y conserva el protocolo; no debe sobrescribir manualmente los runs anteriores.
