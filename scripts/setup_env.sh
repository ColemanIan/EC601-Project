#!/bin/bash -l
# Build or update the cage environment. Run from the repository root.
set -eo pipefail                     # not -u: conda's activation scripts read unset variables
module purge
module load miniconda
if conda env list | grep -q "^cage "; then
    conda env update -n cage -f environment.yml --prune
else
    conda env create -f environment.yml
fi
conda activate cage
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR=/projectnb/ece601/colemian/.cache/pip

python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA:-cu126}"
python -m pip install -e .
mkdir -p requirements
python -m pip freeze --exclude-editable > requirements/lock-scc.txt
python scripts/check_env.py