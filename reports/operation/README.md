# Evidencia de operación P1

La verificación usa el modelo real original de P1, sin reentrenarlo:

- local-compose.json: salud, predicción HTTP y versión READY del Registry de Compose.
- local-compose-restart.json: misma identidad y predicciones después del reinicio.
- verification.json: revisión de código/imagen, arquitectura, estados y pruebas locales.
- clean-compose.json: build y arranque desde una copia del commit sin modelo local, recuperado por HTTP de un Registry accesible.
- architecture.png: orden de arranque y preparación.

Estas comprobaciones funcionales no son pruebas de carga ni mediciones FinOps. Los artifacts de CI se generan con un modelo sintético explícito y se mantienen separados. Las identidades del entrenamiento original y de la importación se distinguen mediante source_run_id; el run nuevo no representa un nuevo entrenamiento.
