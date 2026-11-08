# P1: recuperación del modelo y API FastAPI

Este bloque implementa los puntos 1 y 2 de la entrega. El servicio pronostica
viajes iniciados por estación de Zapopan para la siguiente semana, a partir de
variables ya calculadas. No recibe viajes individuales ni calcula inventario.

## Modelo y procedencia

Modelo local verificado: `mibici-equipo13-p1-demanda`, versión `1`, HGB Poisson
con 7 hojas. Procede del run `65ed8a9510984cf2aedde1663050440a`, elegido por MAE
de validación en la búsqueda `8f09f3642a404fbabff4572ec245b5ea`.
Commit de entrenamiento: `33f9d899fd27e173624c615a808b9c629a627cfe`.
Versión de datos:
`6622c0fec77e6e66f2bdb91f5761af0a792c7d3ce19591d35afc2f842a2ff9a3`.

La versión numérica pertenece a cada Registry. En otra computadora puede ser
distinta; se debe usar la devuelta por el registro de su propia corrida.
El nombre sugerido en un handoff o una URI `models:/m-...` de un modelo guardado
no acredita, por sí solo, su registro por nombre y versión.

`mibici.serving` registra una corrida P1 finalizada y seleccionada por validación,
conservando descripción, corrida, commit y versión de datos. Repetir el registro
de esa misma corrida bajo el mismo nombre reutiliza su versión. No asigna
automáticamente un alias de producción.

## Recuperar una versión existente

Con el entorno del README activado y acceso al Registry que contiene el modelo:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5013
python -m mibici.serving download --name mibici-equipo13-p1-demanda --version 1 --output models/serving/p1
```

En PowerShell se configura la conexión con:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5013"
```

También se admite `--alias NOMBRE` en lugar de `--version`. El alias debe existir;
se resuelve una vez y el paquete conserva la versión numérica concreta.
Un destino existente produce un error: para otra versión utiliza otro directorio
y actualiza `MIBICI_MODEL_DIR`.

El paquete incluye `model/` y `registry.json`, con procedencia y hashes SHA-256.
Se verifica su integridad antes de cargarlo. Los hashes detectan modificaciones
accidentales; el paquete debe proceder de un Registry de confianza.
La API usa ese paquete local y puede arrancar sin conexión a MLflow.
Esta es la opción del enunciado que permite incorporar a la imagen el modelo
previamente descargado del Registry. La imagen se construirá con el bloque de
contenedores del equipo.

## Reproducir desde un clon sin los modelos locales

Las bases y artefactos no están en Git. Si el modelo solo existe en el Registry
de otro integrante, se necesita acceso a ese servidor y a sus artefactos, o
reproducirlo localmente siguiendo estos pasos.

En una terminal con el entorno activado:

```bash
python -m mibici.project server
```

En otra terminal activa el mismo entorno, configura `MLFLOW_TRACKING_URI` y ejecuta:

```bash
mlflow run . --env-manager=local --experiment-name mibici-equipo13 --run-name DATA_reproduce -e ingest
mlflow run . --env-manager=local --experiment-name mibici-equipo13 --run-name P1_search -e train -P question=p1 -P model=hgb-search
python -m mibici.serving register --handoff reports/delivery2/p1/registry_handoff.json
```

La búsqueda debe completarse antes de registrar su handoff: ese archivo se
actualiza con el modelo seleccionado y los identificadores de la nueva ejecución.
Alternativamente, el registro admite `--source models:/ID_DEL_MODELO`.
Descarga después la versión que devuelve el registro mediante el comando de la
sección anterior. La preparación no requiere reentrenar si ya se dispone de una
versión registrada y accesible.

## Ejecutar y consumir la API

```bash
python -m uvicorn mibici.api:app --host 127.0.0.1 --port 8013
curl --fail-with-body http://127.0.0.1:8013/health
curl --fail-with-body -H "Content-Type: application/json" --data-binary @examples/p1-predict.json http://127.0.0.1:8013/predict
```

En Windows usa `curl.exe`. Documentación interactiva: http://127.0.0.1:8013/docs.
`MIBICI_MODEL_DIR` permite otra ruta al paquete. En un contenedor, el proceso
Uvicorn deberá escuchar en `0.0.0.0`; el bloque de contenedores definirá el mapeo
de puerto y la ruta interna al modelo.

El modelo se carga durante el arranque. Si falta el paquete o su integridad es
incorrecta, el servicio no inicia. Las peticiones reutilizan esa instancia.

Petición `POST /predict`:

```json
{
  "instances": [
    {
      "lag_1": 100,
      "mean_previous_4": 95,
      "lag_52": null,
      "week_sin": 0.5,
      "week_cos": 0.8660254037844386
    }
  ]
}
```

| Campo | Significado |
|---|---|
| lag_1 | Viajes iniciados durante la semana anterior |
| mean_previous_4 | Media de las cuatro semanas anteriores |
| lag_52 | Viajes 52 semanas antes; null si no hay esa historia |
| week_sin, week_cos | Calendario de la semana objetivo según candidate_features: fase 2π × (día del año − 1) / 365.2425 |

Cada fila representa una estación y una semana objetivo; se conserva el orden.
Los predictores son exactamente los del entrenamiento, no incluyen station_id
como variable del modelo. El cliente construye las variables con observaciones
anteriores al origen semanal y calendario conocido.

Respuesta real de ejemplo para la primera fila:

```json
{
  "predictions": [97.32063540421066],
  "model": {
    "name": "mibici-equipo13-p1-demanda",
    "version": "1",
    "run_id": "65ed8a9510984cf2aedde1663050440a",
    "git_commit": "33f9d899fd27e173624c615a808b9c629a627cfe",
    "data_version": "6622c0fec77e6e66f2bdb91f5761af0a792c7d3ce19591d35afc2f842a2ff9a3"
  }
}
```

Las predicciones son valores esperados, por lo que pueden ser decimales.
La identidad corresponde al modelo de entrenamiento. El JSON de
`examples/p1-predict.json` incluye dos filas; su respuesta completa está en
[la verificación HTTP](../reports/delivery4/p1-api-verification.json).

Se usa inferencia síncrona por petición, con hasta 1000 filas, para consultar
estaciones cuando ya están disponibles sus variables semanales. No supone
disponibilidad en tiempo real de inventario ni actualiza el modelo con cada
consulta. El esquema básico rechaza estructuras incompletas, campos extra y
valores no finitos. El punto 3 del equipo ampliará y documentará las validaciones
de dominio (rangos, consistencia del calendario y reglas de uso).

## Pruebas y coordinación

```bash
python -m pytest tests/test_api.py -q -W error::DeprecationWarning
python -m pytest -q -W error::DeprecationWarning
```

Las pruebas crean un Registry temporal con un modelo sintético, lo recuperan
por versión y alias, verifican integridad y carga sin red, y consultan Uvicorn
mediante HTTP real con puertos locales temporales. No usan el Registry personal
ni requieren Docker. Hay una excepción de advertencia específica para el uso de
noload de MLflow 3.17 con SQLAlchemy 2.1; las demás deprecaciones siguen tratándose
como errores.

El Containerfile, su construcción/ejecución y la captura exigida desde el
contenedor se integran con los puntos 4–6 del equipo. La verificación HTTP
publicada aquí corresponde a ejecución local sin contenedor.

No se suben modelos, bases, datos ni carpetas de tracking al repositorio.
El contexto del build deberá incluir únicamente el paquete preparado y el
código necesario, excluyendo secretos y datasets.

Fuentes técnicas: [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/workflow/),
[FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/).

## Estado de la integración externa (8 de noviembre de 2026)

La ejecución [42536585381](https://github.com/salinasromerodaniel/TC5061-Proyecto-Equipo13/actions/runs/42536585381)
aprobó 113 pruebas en cada sistema (Windows, Linux y macOS), incluida la API,
y falló únicamente en `test_published_station_catalog_bytes`: la URL histórica
`nomenclatura_2026_09.csv` responde HTTP404. Hay seis pruebas opcionales omitidas.

El [portal oficial](https://www.mibici.net/es/datos-abiertos/) enlaza ahora el
catálogo de octubre, cuyos bytes no coinciden con el hash histórico. Se conserva
la versión de datos anterior. La recuperación de una versión ya registrada y
el paquete de inferencia funcionan con independencia de esa descarga.

Para reconstruir el dataset desde cero mientras falte el catálogo histórico,
se requiere una copia de septiembre verificable con el SHA-256 del manifiesto:
`206e45fe633922b634d858142fa21e6fd9f4c421b6435c02980ef290cc2df90d`.
La copia local existente se verificó. Las pruebas y los hashes originales se
mantienen; el resultado global de CI requiere resolver esa disponibilidad
externa. Detalle en [validation.json](../reports/delivery4/validation.json).
