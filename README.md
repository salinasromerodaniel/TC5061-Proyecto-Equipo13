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

Ingesta implementada y probada con septiembre de 2026: 490,056 filas originales, 758 duplicados operativos removidos y **489,298 viajes limpios**, 468 estaciones utilizadas y 484 registros en el catálogo. No hubo IDs de estación ausentes del catálogo.

Esta ejecución verifica el bloque de ingesta; un mes todavía no cubre el alcance de millones de registros ni fija el periodo final del proyecto. Conservamos toda la red: el filtro de Zapopan, la partición temporal y la aproximación de riesgo de P2 siguen pendientes. Variables, modelos, MLflow y notebook exploratorio siguen sin implementar; los módulos correspondientes aún son ejemplos.

## Estructura

```text
mibici/
  config.py             # Configuración y rutas
  dataset.py            # Ingesta, verificación y reproducción
  features.py           # Pendiente: variables reutilizables
  modeling/
    train.py            # Pendiente: baselines y MLflow
    predict.py          # Plantilla
  plots.py              # Plantilla de visualización
data/                   # Datos locales, excluidos de Git
models/                 # Modelos locales, excluidos de Git
notebooks/              # Exploración
references/             # Manifiesto; pendientes ML Canvas y arquitectura
reports/figures/         # Figuras
```

`data/` está excluido de Git y no aparece en el clon inicial; la ingesta crea sus directorios automáticamente. Los notebooks usarán nombres como `1.0-dsr-exploracion.ipynb`. La lógica final debe estar en el módulo `mibici`.

## Instalación

Python **3.12** (validado con 3.12.15). Instalación, 24 pruebas y reconstrucción de hashes verificadas desde un clon limpio en macOS ARM64; detalle en [references/VALIDACION_INGESTA.md](references/VALIDACION_INGESTA.md). Desde un clon del repositorio y la rama de este cambio:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -e .
```

En Windows puede usarse `py -3.12 -m venv .venv` y activar con `.venv\Scripts\Activate.ps1` en PowerShell. La instalación Windows no se ha validado. No hace falta uv para ejecutar el proyecto.

`requirements.txt` fija dependencias directas y transitivas del bloque de ingesta y pruebas. `requirements.in` declara los requisitos de entrada. Los paquetes de modelado, visualización y notebooks se incorporarán cuando se implementen esos bloques. Para actualizar el archivo de versiones, desde un entorno con uv 0.13.1:

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

Experimento `mibici-equipo13`; al menos un run descriptivo por P1/P2/P3 ejecutado desde código. Registrar rango de fechas, versión de datos, semilla, modelo e hiperparámetros, métricas con conjunto evaluado, artefactos y etiquetas de equipo/pregunta/commit. No registrar datos personales ni credenciales. La integración aún está pendiente.

## Equipo y acceso

- Daniel Salinas Romero: creación del repositorio, estructura inicial e implementación del bloque de ingesta.
- José Alberto Velázquez Trujillo: integrante identificado en el documento colaborativo; rol por confirmar.
- Integrantes restantes y roles: pendientes de confirmación.

El repositorio es público: compañeros y profesor pueden verlo y clonarlo sin invitación. Para subir ramas al mismo repositorio, falta invitar a los compañeros como colaboradores con sus usuarios de GitHub.

## Procedencia

Plantilla oficial: [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/). Opciones de generación en `references/scaffold-config.json`.
