PYTHON_INTERPRETER ?= python

.PHONY: help requirements data reproduce verify test
help:
	@echo "requirements: instalar dependencias; data: reconstruir datos verificados; reproduce: reconstruir septiembre; verify: verificar; test: pruebas"

requirements:
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install --no-build-isolation --no-deps -e .

data reproduce:
	$(PYTHON_INTERPRETER) -m mibici.dataset reproduce

verify:
	$(PYTHON_INTERPRETER) -m mibici.dataset verify

test:
	$(PYTHON_INTERPRETER) -m pytest -q
