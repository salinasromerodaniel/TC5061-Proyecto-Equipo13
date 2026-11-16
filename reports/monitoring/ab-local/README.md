# P1 · Entrenamiento alternativo y comparación A/B de octubre de 2026

B se entrenó con las 5,927 observaciones originales. Se probaron dos
variantes HGB Poisson con los mismos hiperparámetros de P1 salvo el máximo
de hojas. En las 609 filas de validación histórica, la de 15 hojas obtuvo
un MAE de 14.67 y la de 31 hojas, de 15.08. La primera se registró como
versión 2 de `mibici-equipo13-p1-demanda` en el mismo Registry local que A.
Octubre quedó reservado hasta completar la selección y el registro.

A fue la versión 1 de la entrega Docker (run
`01ec9f77835b4cf38601ef75cfd2375f`). Esta reproducción usa las mismas
variables, configuración de siete hojas y datos históricos del P1
documentado. Al volver a ajustar esa configuración obtuvimos exactamente
las mismas predicciones que el artefacto local en las 7,160 filas
históricas. En el resultado se distinguen su linaje y el de la corrida
original que figura en el contrato.

Evaluamos las versiones registradas sobre las mismas 234 observaciones
de 78 estaciones en las semanas del 5, 12 y 19 de octubre. El MAE y el
RMSE se expresan en viajes por estación y semana. Un valor menor
indica mejor desempeño en las tres métricas.

| Modelo | Versión | MAE | RMSE | WAPE |
|---|---:|---:|---:|---:|
| A · HGB Poisson de 7 hojas | 1 | 23.62 | 39.66 | 13.94 % |
| B · HGB Poisson de 15 hojas | 2 | 23.94 | 40.06 | 14.13 % |

**Recomendación: conservar A como versión vigente.** B redujo el MAE
el 12 de octubre, pero aumentó el error en las otras dos semanas y
su MAE global fue 1.36 % mayor. Con tres semanas y estaciones repetidas,
la diferencia es descriptiva. Para decidir si conviene reentrenar,
sigamos midiendo el error observado de A en semanas nuevas y reservemos
un periodo posterior para evaluar cualquier candidato actualizado antes
de promoverlo.

Evidencia: [selección histórica](validation.json), [métricas globales](comparison.csv),
[MAE semanal](weekly.csv) y [resultado con el linaje de MLflow](results.json).
