# Revision de Seguridad

## Manejo de Secrets

Se revisaron los archivos versionados mediante:

```bash
git ls-files
```

No se encontraron credenciales, contraseñas, tokens ni archivos de configuración sensibles.

El proyecto utiliza `.gitignore` para excluir:

- .env
- .env.*
- .venv/
- mlruns/
- mlartifacts/
- *.sqlite
- *.pickle
- *.joblib
- *.parquet

## Analisis Estatico

Se ejecutó:

```bash
bandit -r mibici
```

Resultado:

- Low: 9
- Medium: 0
- High: 0

Los hallazgos corresponden al uso controlado de `subprocess` para operaciones internas de Git y MLflow.

No se detectaron vulnerabilidades de severidad media o alta.

## Auditoria de Dependencias

Se ejecutó:

```bash
pip-audit
```

Resultado:

- 19 vulnerabilidades conocidas
- 4 paquetes afectados

Dependencias afectadas:

- certifi
- mlflow
- pip
- pyasn1

Las versiones corregidas reportadas por `pip-audit` se documentaron como acciones de remediación futuras.

## Conclusion

No se detectó exposición de secretos ni vulnerabilidades de severidad media o alta en el código del proyecto.

Se identificaron dependencias con vulnerabilidades conocidas que deberán actualizarse antes de un despliegue productivo.