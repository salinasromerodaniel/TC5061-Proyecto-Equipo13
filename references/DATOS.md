# Datos e ingesta — primera prueba reproducible

## Fuente y alcance

Portal oficial: https://www.mibici.net/es/datos-abiertos/ (consulta 17 de octubre de 2026).

- Viajes septiembre 2026: https://www.mibici.net/site/assets/files/131825/datos_abiertos_2026_09.csv
- Catálogo septiembre 2026: https://www.mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv
- Archivos, tamaños, hashes, versiones, esquemas y cifras: `data-manifest-2026-09.json`.

Septiembre es una prueba de la base técnica, no la selección final del periodo de entrenamiento. No se ha definido partición train/validación/test ni se ha filtrado Zapopan. El catálogo no tiene una columna de municipio y contiene registros `ex-`; sus nombres no prueban por sí solos pertenencia municipal ni vigencia histórica. La capacidad actual tampoco demuestra la capacidad de meses anteriores.

## Reglas de procesamiento

- CSV descargado por HTTPS con verificación TLS, timeout y reintentos para errores transitorios. Escritura temporal y reemplazo por archivo completo; no se aceptan fuentes cuyo hash difiera del manifiesto de reconstrucción.
- Codificación validada sobre todo el archivo: UTF-8 con BOM opcional, después CP1252 y Latin-1. Separador detectado entre coma, punto y coma o tabulador. Encabezados normalizados en mayúsculas, espacios y acentos; columnas requeridas ausentes o ambiguas detienen el proceso.
- Salida de viajes: `trip_id`, `started_at`, `ended_at`, `origin_station_id`, `destination_station_id`. Los IDs deben ser enteros positivos; fechas válidas, fin no anterior a inicio y fecha de inicio dentro del mes del archivo.
- Fechas ISO y día/mes/año explícitos admitidos; no se infiere zona horaria ni se convierten tiempos. La interpretación temporal debe confirmarse antes de feature engineering.
- Para un mismo mes, ID repetido con los mismos campos operativos: conservar primera copia y contar remociones, incluso entre lotes. ID repetido con campos operativos distintos: detener, sin escoger arbitrariamente un viaje. No se verifica unicidad de IDs entre distintos meses.
- Catálogo: ID único positivo; nombre no vacío, latitud/longitud dentro de rangos válidos y capacidad entera no negativa. Se conserva el catálogo completo, incluyendo nombres `ex-` y capacidades cero; no equivale a un inventario de estaciones activas.
- Estaciones de viaje no encontradas en el catálogo se reportan en el manifiesto, sin descartar viajes ni inferir coordenadas o capacidad.
- No se descartan otras anomalías silenciosamente. En caso de error, se devuelve código distinto de cero y no se publica un manifiesto nuevo. Pueden quedar archivos completos de pasos anteriores; solo un manifiesto verificado acredita el conjunto.
- Parquet con Zstandard, esquema explícito y procesamiento por lotes. El índice de deduplicación crece con los IDs de un mes y se libera al pasar al siguiente.

## Resultado de septiembre de 2026

| Medida | Resultado |
|---|---:|
| Filas originales | 490,056 |
| Filas duplicadas removidas | 758 |
| Viajes limpios | 489,298 |
| Estaciones utilizadas en viajes | 468 |
| Registros del catálogo | 484 |
| IDs usados ausentes del catálogo | 0 |

Los CSV originales contienen 758 duplicados completos; la limpieza operativa obtiene la misma cantidad removida. El mes no alcanza por sí solo el alcance de millones de registros requerido por el entregable.

## Datos personales y condiciones de uso

El CSV original incluye identificador de usuario, género y año de nacimiento. Estos campos se excluyen de Parquet porque no se requieren para las tres preguntas propuestas. Los originales permanecen locales y fuera de Git; no subir muestras de filas a reportes, parámetros o artefactos. El identificador de viaje se conserva para integridad y deduplicación, sin asociarlo a datos de usuario en la tabla limpia.

En la página consultada se ofrecen descargas abiertas; no se identificó allí una licencia específica de reutilización del dataset. No afirmar que usa Creative Commons ni confundir los términos de suscripción del servicio con una licencia de datos. Falta confirmar y documentar las condiciones de reutilización para el PDF; atribuir siempre la fuente MiBici.

## Versionado

El manifiesto versionado contiene fuentes exactas, filas originales y limpias, duplicados, esquemas, rango observado, versiones de pandas/pyarrow y SHA-256 de CSV y Parquet. `data_version` es el SHA-256 de la representación JSON canónica de todo el manifiesto, excluyendo ese mismo campo. No incluye hora de ejecución ni rutas absolutas.

`verify` comprueba hashes y conteos locales sin red. `reproduce` verifica la versión de transformación y dependencias de serialización, recupera las fuentes exactas y exige reconstruir el mismo manifiesto. Una actualización del portal puede impedir reconstruir una versión si elimina o reemplaza el original; los hashes detectan el cambio, pero no sustituyen un archivo histórico externo. No se ha configurado ese almacenamiento adicional.

## Transformación 2 — coordenadas consistentes entre entornos

El 18 de octubre de 2026 se detectaron diferencias en los últimos bits de `latitude` entre macOS y los catálogos reconstruidos por dos compañeros. Filas, IDs, nombres, longitudes y capacidades coincidían. La diferencia máxima frente a la conversión de texto mediante `float` era de 3.552713678800501e-15 grados. El hash binario correctamente rechazó esos resultados.

La transformación 2 usa `float` de Python sobre cada coordenada del CSV y almacena `float64`; rechaza valores ausentes, no numéricos, no finitos o fuera de rango. Se aplica a latitud y longitud para definir una sola regla de conversión, sin reducir precisión mediante redondeo. Referencia del conversor: https://docs.python.org/3.12/library/functions.html#float.

Los manifiestos `-v2.json` conservan fuentes, conteos y hashes de todos los viajes; cambian la transformación, el hash/tamaño del catálogo y la versión global de datos. Los manifiestos anteriores se conservan sin cambios para trazabilidad de los runs históricos. Usar una versión antigua contra un catálogo ya reparado producirá una discrepancia esperada.

`repair-stations --manifest <manifiesto-v2>` comprueba todos los demás archivos y reconstruye únicamente el catálogo, verificando el candidato antes del reemplazo. No salta la integridad de viajes ni altera el manifiesto. `reproduce` con v2 permite una reconstrucción completa desde las fuentes exactas.
