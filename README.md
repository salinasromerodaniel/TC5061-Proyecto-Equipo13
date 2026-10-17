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

Repositorio generado con Cookiecutter Data Science 2.3.0 y `include_code_scaffold=Yes`. Los módulos son plantillas; aún no implementan ingesta, variables ni experimentos reales. Sus mensajes de ejemplo no prueban procesamiento ni entrenamiento.

Pendientes: fechas y fuentes verificadas, partición temporal, fórmula de P2, dependencias fijadas, ingesta automática y manifiesto, exploración, variables reutilizables, tres baselines en MLflow, ML Canvas, arquitectura y PDF. La instalación desde un clon limpio todavía no está validada.

## Estructura

```text
mibici/
  config.py             # Configuración y rutas
  dataset.py            # Pendiente: ingesta y versionado
  features.py           # Pendiente: variables reutilizables
  modeling/
    train.py            # Pendiente: baselines y MLflow
    predict.py          # Plantilla
  plots.py              # Plantilla de visualización
data/                   # Datos locales, excluidos de Git
models/                 # Modelos locales, excluidos de Git
notebooks/              # Exploración
references/             # ML Canvas, arquitectura y procedencia de plantilla
reports/figures/         # Figuras
```

`data/` está excluido de Git y no aparece en el clon inicial; la futura ingesta deberá crear sus directorios. Los notebooks usarán nombres como `1.0-dsr-exploracion.ipynb`. La lógica final debe estar en el módulo `mibici`.

## Entorno previsto

Python 3.12, según `pyproject.toml`. Cuando el equipo disponga de ese intérprete:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En PowerShell, activar con `.venv\Scripts\Activate.ps1`. Las dependencias de la plantilla aún no están fijadas; estos comandos describen el arranque previsto y no acreditan un entorno reproducible validado.

No hay todavía comandos funcionales de ingesta o experimentos base. Se documentarán aquí cuando se implementen; no usar los scripts de ejemplo como evidencia de ejecución del entregable.

## Colaboración

Seguimos GitHub Flow: crear una rama desde `main`, trabajar en ella, abrir un Pull Request y solicitar revisión de otro integrante antes de integrar. Tras el primer commit de la plantilla, los cambios deben pasar por PR revisados. La protección de `main` está activada: exige un PR con al menos una aprobación, descarta aprobaciones tras cambios y aplica también al administrador. Se bloquean force-push y eliminación de la rama.

- Ramas: `feat/descripcion`, `fix/descripcion`, `docs/descripcion`, `chore/descripcion`.
- Commits: Conventional Commits, por ejemplo `feat: add monthly trip ingestion`.
- Cada integrante debe aportar commits, abrir al menos un PR y revisar al menos uno de otro compañero.
- No subir datos, carpetas de MLflow, modelos ni credenciales. Mantener configuración, fechas y semillas en `mibici/config.py`.

## Buenas prácticas previstas de MLflow

Experimento `mibici-equipo13`; al menos un run descriptivo por P1/P2/P3 ejecutado desde código. Registrar rango de fechas, versión de datos, semilla, modelo e hiperparámetros, métricas con conjunto evaluado, artefactos y etiquetas de equipo/pregunta/commit. No registrar datos personales ni credenciales. La integración aún está pendiente.

## Equipo y acceso

- Daniel Salinas Romero: creación del repositorio y base inicial de estructura/documentación.
- José Alberto Velázquez Trujillo: integrante identificado en el documento colaborativo; rol por confirmar.
- Integrantes restantes y roles: pendientes de confirmación.

El repositorio es público: compañeros y profesor pueden verlo y clonarlo sin invitación. Para subir ramas al mismo repositorio, falta invitar a los compañeros como colaboradores con sus usuarios de GitHub.

## Procedencia

Plantilla oficial: [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/). Opciones de generación en `references/scaffold-config.json`.
