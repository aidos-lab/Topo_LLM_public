#!/bin/bash

set -e

# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
    echo "Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
    exit 1
fi

echo "📂 TOPO_LLM_REPOSITORY_BASE_PATH=${TOPO_LLM_REPOSITORY_BASE_PATH}"

echo "⏳ Loading environment variables from .env file..."
source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

export PYTORCH_ENABLE_MPS_FALLBACK=1
SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/"
SCRIPT_PATH+="topollm/analysis/distribution_over_tokens/submit_analyse_local_estimate_distribution.py"

# # # # # # # # # # # #
LAUNCHER="basic"
RUN_MODE="dry_run"

# LAUNCHER="hpc_submission"
# RUN_MODE="regular"
# # # # # # # # # # # #

echo "⏳ Starting submission script..."
echo "📂 Script path: $SCRIPT_PATH"

uv run $SCRIPT_PATH \
    --launcher $LAUNCHER \
    --run-mode $RUN_MODE

echo "✅ Submission script completed successfully."
echo "✅ Exiting."
exit 0
