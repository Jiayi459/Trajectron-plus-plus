#!/bin/bash
# =============================================================================
# One-time environment setup for Trajectron++ on Notre Dame CRC (UGE cluster)
# Run this ONCE on a front-end node after cloning the repo:
#     ssh netid@crcfe01.crc.nd.edu
#     cd ~/Trajectron-plus-plus
#     bash crc/setup_env.sh
#
# It creates a conda env named `trajectron++` with a CUDA-enabled PyTorch.
# =============================================================================
set -euo pipefail

ENV_NAME="trajectron++"
PY_VERSION="3.10"

echo "=== Loading conda module ==="
# NOTE: verify the exact module name with `module avail conda` / `module avail python`.
# ND CRC commonly exposes conda via one of these; the script tries `conda` then `python`.
module load conda 2>/dev/null || module load python 2>/dev/null || true

# Make `conda activate` work inside a non-interactive shell.
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== Creating conda env: $ENV_NAME (python $PY_VERSION) ==="
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "$ENV_NAME" "python=${PY_VERSION}"
fi
conda activate "$ENV_NAME"

echo "=== Installing CUDA-enabled PyTorch 1.13.1 (cu117 wheel) ==="
# cu117 supports V100 (sm_70) and A10 (sm_86), the common CRC general-access GPUs.
# If you are scheduled onto a newer GPU (H100/sm_90), you must upgrade torch + CUDA.
pip install --upgrade pip
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

echo "=== Installing the rest of requirements ==="
# torch==1.13.1 is already satisfied by the +cu117 build above.
pip install -r requirements.txt

echo "=== Pinning setuptools so ncls can import pkg_resources ==="
pip install setuptools==75.8.2

echo "=== Verifying CUDA is visible to torch (run on a GPU node to see True) ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available (False is expected on a front-end node):", torch.cuda.is_available())
PY

echo ""
echo "=== DONE. Env '$ENV_NAME' ready. ==="
echo "Next: bash crc/process_data.job is a qsub script -> run:  qsub crc/process_data.job"
