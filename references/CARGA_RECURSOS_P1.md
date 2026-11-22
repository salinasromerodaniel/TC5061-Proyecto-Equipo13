# Pruebas de carga y recursos de P1

Los puntos 3 y 4 de la actividad usan la API síncrona y el servidor MLflow
del Compose existente. La API carga el modelo una vez al arrancar y atiende
POST /predict. Cada petición contiene las dos instancias del ejemplo público.
Locust valida HTTP 200, dos predicciones finitas no negativas y la identidad
del modelo que respondió /health antes de medir.

## Preparación en WSL o Linux

Activa Docker y prepara un paquete real siguiendo [OPERACION_P1.md](OPERACION_P1.md).
Desde la raíz del repositorio, configura la fuente y el SHA-256 reales del modelo
en .env. Si usas la reproducción de la entrega Docker, su SHA-256 es
`bd8bb2815437b744a33a92628fc61452332a4d37763cc8a1b723b62aaf4897ae`.
Conserva la procedencia exacta que devuelve /health. El número de versión
del Registry pertenece a cada servidor.

```bash
docker compose -p mibici-load-e13 up --build -d --wait --wait-timeout 240
python3.12 -m venv data/load-env
data/load-env/bin/python -m pip install -r requirements-load.txt
data/load-env/bin/python -m pip check
data/load-env/bin/python scripts/run_load.py --project mibici-load-e13 \
  --users 1 10 50 --seconds 90 --output reports/load/mi-ejecucion
data/load-env/bin/python scripts/summarize_load.py reports/load/mi-ejecucion
```

El runner necesita Docker CLI/Compose y acceso a los dos puertos publicados.
Para otros puertos, usa --api-url y --tracking-uri. --project identifica
el mismo proyecto de Compose que está levantado. El destino debe ser nuevo
para conservar cada ejecución. En un checkout normal, Git registra la revisión
del código automáticamente. Si se usa un worktree creado desde Windows en WSL,
Git necesita GIT_DIR y GIT_WORK_TREE con rutas Linux al worktree y su metadato.

## Protocolo

Se observan 20 segundos de reposo con ambos servicios preparados y se ejecuta
un calentamiento separado de 15 segundos con dos usuarios. A continuación,
cada escenario dura 90 segundos y escala hasta su concurrencia en aproximadamente
cinco segundos (un usuario llega de inmediato). Se incluye la rampa en las
métricas y se dejan cinco segundos entre escenarios.

El modelo de carga es cerrado y cada usuario vuelve a pedir una predicción
al terminar la anterior, sin pausa deliberada. Un timeout de diez segundos
marca fallos y se conceden quince segundos para terminar peticiones al cierre.
El generador usa HttpUser y conexiones persistentes. Un fallo de validación
del cuerpo también cuenta como error. Este ensayo describe el lote fijo de
dos instancias, bajo concurrencia continua. Otro tamaño de lote o perfil
de tráfico requiere una ejecución nueva.

docker stats --no-stream recoge los dos contenedores en cada consulta,
con una pausa de 0.2 segundos entre consultas. Docker tarda aproximadamente
dos segundos en calcular cada muestra. Se guardan los objetos JSON originales
y la hora de recepción; la frecuencia real queda visible en resources.csv.
El generador Locust se mide con psutil cada dos segundos, por separado.
Un consumo de 100 % de CPU equivale a un núcleo, y coincide con el límite
de un CPU configurado para cada servicio. La memoria de docker stats en
Linux descuenta caché de archivos según las reglas de Docker. Los límites
de 512 MiB para API y 1 GiB para MLflow son cotas, no consumo observado.

## Evidencia y lectura

Cada escenario conserva el HTML nativo de Locust y sus CSV de estadísticas,
historia, fallos y excepciones. La latencia media proviene de Locust y los
percentiles usan sus histogramas redondeados. La serie de p95 y RPS del CSV
histórico usa ventanas móviles, mientras la tabla final resume el escenario
entero. La gráfica de errores usa fallos por segundo divididos entre
peticiones por segundo en la misma muestra. Un valor cero se interpreta
como ningún fallo observado en esa ejecución.

El resumen distingue peticiones/s de inferencias exitosas/s. Para este lote,
inferencias exitosas/s = peticiones/s × (1 − tasa de fallos) × 2.
resource-summary.csv muestra medias y máximos muestreados, incluido el reposo.
Los intervalos de recursos abarcan el proceso Locust de cada escenario,
incluido su arranque y cierre. metadata.json conserva el modelo, las versiones
de herramientas, los hashes de los scripts y el estado de los contenedores
antes y después, con indicadores de OOM y reinicios.

Locust y los servicios comparten la máquina WSL. El ensayo permite comparar
estos escenarios locales, con una sola repetición y orden creciente. La red,
el host y el generador también influyen. La evolución de throughput, latencia,
CPU y memoria ayuda a localizar restricciones. Para establecer capacidad o
un SLO se requieren repeticiones y un generador en otra máquina, junto con
el objetivo de latencia que adopte el equipo.

Las tablas aportan demanda de CPU, memoria y throughput al responsable
del punto 5. La estimación de costo debe sumar los componentes que decida
incluir y fijar proveedor, región, tarifa y utilización. Las mediciones
locales no fijan por sí solas un precio de nube.

Fuentes técnicas: [ejecución headless de Locust](https://docs.locust.io/en/stable/running-without-web-ui.html),
[salidas CSV y HTML](https://docs.locust.io/en/stable/config-options.html),
[métricas de docker stats](https://docs.docker.com/reference/cli/docker/container/stats/).
