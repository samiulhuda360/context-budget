# One command to set up, one to run the demo.
PY ?= python

setup:
	$(PY) -m pip install -e ".[dev]"

demo:
	ctxb demo

serve:
	ctxb serve --upstream mock

check:
	ruff check .
	ruff format --check .
	mypy .
	pytest
	ctxb bench --check
	ctxb eval offline --check
	ctxb eval model --offline --partial --check --quiet

.PHONY: setup demo serve check
