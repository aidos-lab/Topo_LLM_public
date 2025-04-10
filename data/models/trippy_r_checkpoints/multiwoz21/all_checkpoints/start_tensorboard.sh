#!/bin/bash

REPOSITORY_BASE_PATH_VARIABLE_NAME="TOPO_LLM_REPOSITORY_BASE_PATH"

# Check if the environment variable with the name $REPOSITORY_BASE_PATH_VARIABLE_NAME is set
if [ -z "${!REPOSITORY_BASE_PATH_VARIABLE_NAME}" ]; then
    echo "@@@ The environment variable $REPOSITORY_BASE_PATH_VARIABLE_NAME is not set."
    exit 1
fi

LOGDIR="${!REPOSITORY_BASE_PATH_VARIABLE_NAME}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/runs"

echo ">>> Starting TensorBoard with LOGDIR=${LOGDIR} ..."

uv run tensorboard \
    --logdir "${LOGDIR}"

echo ">>> Finished starting TensorBoard."

# Exit the script
echo ">>> Exiting script."
exit 0
