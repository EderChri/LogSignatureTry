#!/bin/bash

set -euo pipefail

module purge
module load Python/3.13.1-GCCcore-14.2.0
export PYTHONNOUSERSITE=1

VENV="${VENV:-$HOME/PyEnvMvcl-py313}"

if [ ! -d "$VENV" ]; then
  echo "Creating virtualenv: $VENV"
  python -m venv "$VENV"
fi

source "$VENV/bin/activate"
python -m ensurepip --upgrade
python -m pip install --upgrade pip wheel
python -m pip install --upgrade -r requirements.txt
python -m pip install --upgrade -r requirements-idun-torch.txt
python -m pip install --upgrade -r requirements-idun-torch-stack.txt
python -m pip install --no-deps -e .
