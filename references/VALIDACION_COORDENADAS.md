# Validación de coordenadas — transformación 2

Fecha: 18 de octubre de 2026.

## Diferencia observada

Dos compañeros obtuvieron los mismos hashes por columna entre sí, pero la latitud difería de la referencia macOS. El catálogo conservaba 484 filas y las otras cuatro columnas coincidían. La diferencia máxima respecto a `float` del texto original fue 3.552713678800501e-15 grados: 21 valores en Windows y 12 en la referencia macOS.

## Corrección

Conversión explícita del texto de ambas coordenadas mediante `float` de Python, almacenamiento `float64`, validación de valores ausentes, no numéricos, no finitos y fuera de rango. Sin redondeo añadido. Se mantienen los manifiestos de transformación 1 y se publican manifiestos nuevos `-v2.json`.

Catálogo v2: 20,175 bytes, SHA-256 `275c806c8f13cbe2ae040c614e7ba530e2dd8738050565ad3a384016ef749141`.

Datos de 24 meses v2: `6622c0fec77e6e66f2bdb91f5761af0a792c7d3ce19591d35afc2f842a2ff9a3`.

## Validación local

- 51 pruebas unitarias/de integración sintética aprobadas; una prueba de red se ejecutó por separado y pasó.
- El catálogo oficial, descargado y verificado por SHA-256, produce los bytes v2 publicados.
- `repair-stations` completó la comprobación real de todos los originales y Parquet de viajes de los 24 meses y reconstruyó solo el catálogo. No se descargaron ni reprocesaron viajes.
- Comparación espacial con el catálogo anterior: las mismas 106 estaciones de referencia cubiertas por el polígono de Zapopan.
- Los hashes de viajes y manifiestos históricos permanecen iguales. El catálogo reparado requiere verificar con el manifiesto v2.

La ejecución de P1 y la matriz de GitHub Actions se comprobarán antes de cerrar el PR. La presencia de un workflow no constituye por sí sola evidencia de que haya pasado.
