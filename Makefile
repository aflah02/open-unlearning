.PHONY: quality style

check_dirs := scripts src #setup.py

quality:
	ruff check $(check_dirs) setup.py setup_data.py
	ruff format --check $(check_dirs) setup.py setup_data.py

style:
	ruff check $(check_dirs) setup.py setup_data.py --fix
	ruff format $(check_dirs) setup.py setup_data.py

test:
	PYTHONPATH=src CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests
