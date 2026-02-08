PYTHON ?= python3
VENV ?= .venv

.PHONY: venv install lint test smoke full plots clean

venv:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip==25.0.1

install:
	. $(VENV)/bin/activate && pip install '.[dev]'

lint:
	. $(VENV)/bin/activate && ruff check src tests scripts

test:
	. $(VENV)/bin/activate && pytest

smoke:
	. $(VENV)/bin/activate && jssp-yafs smoke --config configs/smoke.yaml

full:
	. $(VENV)/bin/activate && jssp-yafs full --config configs/full.yaml

plots:
	. $(VENV)/bin/activate && jssp-yafs plots --config configs/full.yaml

clean:
	rm -rf .pytest_cache .ruff_cache
