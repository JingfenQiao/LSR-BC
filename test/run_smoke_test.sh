#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=lsr-bc-smoke
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --output=/fnwi_fs/ivi/irlab/personal/jqiao/LSR-BC/log/smoke_test.out
#SBATCH --error=/fnwi_fs/ivi/irlab/personal/jqiao/LSR-BC/log/smoke_test.out

set -e

ENV_NAME="lsr-bc"
PROJECT_DIR="/fnwi_fs/ivi/irlab/personal/jqiao/LSR-BC"

echo "========================================"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date:    $(date)"
echo "========================================"

# ── create conda env if not exists ───────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "[setup] Creating conda env '${ENV_NAME}' ..."
    conda create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"
echo "[setup] Python: $(which python) $(python --version)"

# ── install dependencies ──────────────────────────────────────────────────────
echo "[setup] Installing dependencies ..."
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu118
pip install -q transformers datasets sentence-transformers
pip install -q mteb ir-datasets ir-measures
pip install -q scikit-learn numpy matplotlib tqdm wandb

echo "[setup] Done. Package versions:"
python -c "import torch; print(f'  torch {torch.__version__}, cuda={torch.cuda.is_available()}')"
python -c "import transformers; print(f'  transformers {transformers.__version__}')"
python -c "import mteb; print(f'  mteb {mteb.__version__}')"

# ── run smoke tests ───────────────────────────────────────────────────────────
echo ""
echo "[test] Running smoke tests ..."
cd "${PROJECT_DIR}"
python test/smoke_test.py

echo ""
echo "[done] $(date)"
