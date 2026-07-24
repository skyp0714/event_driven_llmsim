#!/bin/bash

# Launch the simulator Docker container (ASTRA-Sim + sim Python deps).
#
# Mounts the repo root regardless of where this script is invoked from.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../scripts
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                    # .../LLMServingSim

docker run --name servingsim_docker \
  -it \
  -v "$REPO_ROOT":/app/LLMServingSim \
  -w /app/LLMServingSim \
  astrasim/tutorial-micro2024 \
  bash -c "pip3 install pyyaml pyinstrument transformers datasets \
  msgspec scikit-learn xgboost==3.1.2 matplotlib==3.5.3 pandas==1.5.3 \
  numpy==1.23.5 && exec bash"