PY := ./.conda-env/bin/python
PIP := ./.conda-env/bin/pip
CONDA := /opt/conda/bin/conda

.PHONY: help env test lint data target docking model admet figures all clean finalize predict report

help:
	@echo "LGMN inhibition pipeline"
	@echo
	@echo "  make env       create the conda env and install dependencies"
	@echo "  make test      run the test suite"
	@echo "  make data      fetch ChEMBL data and build the modelling set"
	@echo "  make target    survey LGMN structures, pick and prepare a template"
	@echo "  make docking   validate the docking protocol, then screen"
	@echo "  make model     train and evaluate the QSAR models"
	@echo "  make admet     apply ADMET/CNS filters and build the shortlist"
	@echo "  make all       the whole pipeline, in dependency order"
	@echo
	@echo "Docking is the slow step; on two cores the screen takes hours."

env:
	$(CONDA) create -y -p ./.conda-env python=3.12
	$(PIP) install -r requirements.txt
	$(CONDA) install -y -p ./.conda-env -c conda-forge smina openbabel

test:
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m ruff check src tests || true

# --- pipeline stages ------------------------------------------------------

data:
	$(PY) src/data/fetch_chembl.py
	$(PY) src/data/prepare_dataset.py
	$(PY) src/data/fetch_pubchem.py
	$(PY) src/data/explore.py
	$(PY) src/data/permeability.py
	$(PY) src/data/selectivity.py

target:
	$(PY) src/docking/characterize_target.py
	$(PY) src/docking/prepare_receptor.py

docking:
	$(PY) src/docking/validate_protocol.py
	$(PY) src/docking/dock.py
	$(PY) src/docking/analyse_screen.py

model:
	$(PY) src/models/train.py
	$(PY) src/models/train.py --data data/processed/lgmn_combined.csv --tag _combined
	$(PY) src/models/compare_datasets.py

admet:
	$(PY) src/admet/screen.py

report:
	$(PY) src/report.py
	$(PY) src/report_candidates.py

all: data target model docking admet report

clean:
	rm -rf results/docking/poses results/models/*.joblib
	@echo "Kept data/ and the CSV outputs; removed poses and fitted models."

predict:
	$(PY) src/models/predict.py --input $(IN) --output $(OUT)

# Run after the docking screen completes: analyse the scores, fold them into
# the shortlist, and regenerate every write-up from the new results.
finalize:
	$(PY) src/docking/analyse_screen.py
	$(PY) src/admet/screen.py
	$(PY) src/models/compare_datasets.py
	$(PY) src/report.py
	$(PY) src/report_candidates.py
