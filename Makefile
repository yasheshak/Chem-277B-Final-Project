.PHONY: setup install-pyg test clean

setup:
	conda env create -f environment.yml
	@echo "Run: conda activate homo-lumo"
	@echo "Then run: make install-pyg"

install-pyg:
	pip install torch-scatter torch-cluster torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html

test:
	python -c "import torch; from torch_geometric.nn import DimeNetPlusPlus; from torch_scatter import scatter; print('All imports OK')"

clean:
	conda env remove -n homo-lumo -y
