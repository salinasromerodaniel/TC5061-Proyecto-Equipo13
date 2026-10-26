# Parte C — Registro y ciclo de vida de P2

Se registró la regresión logística seleccionada para P2 con el nombre
`mibici-equipo13-p2-riesgo`. La versión 1 tiene el alias `champion`, que identifica
la versión de producción para esta entrega académica. No implica que exista
un sistema operativo de rebalanceo desplegado.

Se eligieron aliases porque permiten mantener un nombre estable para cargar el
modelo aunque cambie la versión seleccionada. Las versiones numeradas conservan
el historial y permiten volver a una versión anterior. No se combinaron aliases
con las etapas clásicas de producción.

El registro incluye una descripción de la tarea y sus límites, la ejecución de
origen, el experimento, la versión del código y la de los datos. Los identificadores
completos se conservan en MLflow y en `reports/delivery2/p2/registry.json`.
Se comprobó la carga por versión y por alias, y se compararon las predicciones y
probabilidades con las del modelo original sobre los tres ejemplos guardados
con él. Coincidieron. Esta comprobación valida la recuperación del modelo;
no sustituye su evaluación sobre el conjunto de prueba documentada en la Parte B.

## 1. Preparar la corrida seleccionada

La Parte B de P2 debe estar integrada, con sus resultados y artefactos completos
en el servidor seleccionado. El archivo `results.json` debe identificar al ganador
y ese servidor debe poder recuperar su modelo y su ejemplo de entrada.

## 2. Registrar y comprobar el modelo

Con el entorno del equipo activo, desde la raíz del repositorio:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5013
python -m mibici.modeling.p2_registry
```

El script usa la selección de `reports/delivery2/p2/results.json`, comprueba su
origen, registra o reutiliza la versión correspondiente y verifica la carga antes
y después de asignar `champion`. Si el alias ya apunta a otra corrida, se detiene
para evitar sustituirla inadvertidamente. No reentrena ni cambia P1/P3.

## 3. Cargar el ganador por su alias

Para cargar el ganador:

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5013")
mlflow.set_registry_uri("http://127.0.0.1:5013")
modelo = mlflow.sklearn.load_model("models:/mibici-equipo13-p2-riesgo@champion")
```

## 4. Identificar el servidor correcto

El registro verificado se encuentra en el tracking original de P2, en la copia
`Documents/Codex/2026-10-24/como-x20/work/TC5061-Proyecto-Equipo13`, servido
localmente por HTTP en el puerto 5014. No está en la base antigua de la copia
`~/TareasMaestria/TC5061-Proyecto-Equipo13`. El puerto 5013 anterior es el ejemplo
normal del equipo: debe apuntar al servidor que realmente dispone de las corridas.
Usar el mismo nombre de experimento no sincroniza bases ni artefactos. Para registrar
en otro servidor hay que reproducir P2 allí o trasladar correctamente el tracking.
Nunca se deben subir SQLite, datos, carpetas de tracking ni modelos a GitHub.

## 5. Revisar la evidencia de la interfaz

La vista del modelo debe mostrar su descripción, versión 1 y alias `champion`.
El detalle de la versión debe permitir verificar la ejecución de origen y las
etiquetas de trazabilidad. Las capturas se incorporan al reporte de P2.

## Referencia

MLflow contributors. (s. f.). Model Registry workflows. Recuperado el 25 de octubre
de 2026, de https://mlflow.org/docs/latest/ml/model-registry/workflow/
