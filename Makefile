.PHONY: setup install-pyg test clean download-data schnet test-schnet dimenet test-dimenet

SCHNET_SCRIPT = ./SchNet/SchNet_Final.py
SCHNET_TEST_SCRIPT = ./SchNet/SchNet_Test.py
DIMENET_SCRIPT = ./DimeNet/DimeNet_Final.py
DIMENET_TEST_SCRIPT = ./DimeNet/DimeNet_Test.py
DOWNLOAD_SCRIPT = download_data.py

setup:
	conda env create -f environment.yml
	@echo "Run: conda activate homo-lumo"
	@echo "Then run: make install-pyg"

install-pyg:
	conda run -n homo-lumo python -m pip install ase-db-backends torch-scatter torch-cluster torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html

test:
	python -c "import torch; from torch_geometric.nn import DimeNetPlusPlus; from torch_scatter import scatter; print('All imports OK')"

clean:
	conda env remove -n homo-lumo -y

download-data:
	python $(DOWNLOAD_SCRIPT)

schnet:
	python $(SCHNET_SCRIPT) $(ARGS)

test-schnet:
	python $(SCHNET_TEST_SCRIPT) $(ARGS)

dimenet:
	python $(DIMENET_SCRIPT) $(ARGS)

test-dimenet:
	python $(DIMENET_TEST_SCRIPT) $(ARGS)
