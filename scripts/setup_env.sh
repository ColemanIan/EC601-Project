#!/bin/bash -l
# Build or update the imoge environment. Run from the repository root.
set -eo pipefail  # if bash fails, exit and return error code

module purge
module load miniconda
if conda env list | grep -q "^imoge "; then
    mamba env update -n imoge -f environment.yml --prune
else
    mamba env create -f environment.yml
fi
conda activate imoge
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR=/projectnb/ece601/colemian/.cache/pip

python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA:-cu126}"
python -m pip install -e .
mkdir -p requirements
python -m pip freeze --exclude-editable > requirements/lock-scc.txt
python scripts/check_env.py