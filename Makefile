VENV   ?= .venv
PYTHON := $(VENV)/bin/python
PIP    := $(PYTHON) -m pip
JUPYTER := $(PYTHON) -m jupyter

export MPLBACKEND := Agg

.PHONY: setup p3 p4 all notebook clean

setup:
	@test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

p3:
	$(PYTHON) proyecto3/tarea1_kmeans.py
	$(PYTHON) proyecto3/tarea2_meanshift.py
	$(PYTHON) proyecto3/tarea3_regresion_lineal.py
	$(PYTHON) proyecto3/tarea4_distancia_coseno.py

p4:
	$(PYTHON) proyecto4/tarea5_centroide_3d.py
	$(PYTHON) proyecto4/tarea6_iris_kmeans.py
	$(PYTHON) proyecto4/tarea7_iris_knn.py
	$(PYTHON) proyecto4/tarea8_iris_centroids.py

all: p3 p4

notebook:
	$(PYTHON) -m notebook

clean:
	rm -rf proyecto3/output proyecto4/output proyecto3/__pycache__ proyecto4/__pycache__ proyecto3/.ipynb_checkpoints proyecto4/.ipynb_checkpoints
