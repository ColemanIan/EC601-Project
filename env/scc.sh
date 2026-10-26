# Usage: source env/scc.sh     (interactive sessions and job scripts)
module purge # clears any module loaded earlier in the session
module load miniconda
conda activate CAGE

export PROJ=/projectnb/ece601/colemian
export CAGE_DATA=$PROJ/data
export CAGE_RUNS=$PROJ/runs
export PYTHONNOUSERSITE=1                    # ignore ~/.local packages entirely
export PIP_CACHE_DIR=$PROJ/.cache/pip        # PyTorch wheels are about 2 GB; keep them out of home
export HF_HOME=$PROJ/.cache/huggingface      # MoGe checkpoints
export TORCH_HOME=$PROJ/.cache/torch         # torchvision COCO weights