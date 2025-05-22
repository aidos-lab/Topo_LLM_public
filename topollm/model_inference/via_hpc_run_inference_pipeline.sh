#!/bin/bash

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

echo "TOPO_LLM_REPOSITORY_BASE_PATH=${TOPO_LLM_REPOSITORY_BASE_PATH}"

PYTHON_SCRIPT_NAME="run_inference_pipeline.py"
RELATIVE_PYTHON_SCRIPT_PATH="topollm/model_inference/${PYTHON_SCRIPT_NAME}"
ABSOLUTE_PYTHON_SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/${RELATIVE_PYTHON_SCRIPT_PATH}"

# ==================================================== #
# Select the parameters here

# LANGUAGE_MODEL_LIST="roberta-base"
# CHECKPOINT_NO_LIST="" # Explicitly unset or empty CHECKPOINT_NO_LIST to avoid interference from environment variables

# # # #
# Define an array of language models and checkpoints
#

# # === START Version 1: Checkpoints from training for 5 epochs
# #

# LANGUAGE_MODEL_ARRAY=(
#   "model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5"
#   "model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5"
# )
# CHECKPOINT_NO_ARRAY=(
#     "400"
#     "800"
#     "1200"
#     "1600"
#     "2000"
#     "2400"
#     "2800"
# )

# #
# # === END Version 1: Checkpoints from training for 5 epochs

# === START Version 2: Checkpoints from training for 50 epochs
#

LANGUAGE_MODEL_ARRAY=(
  "model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50"
  "model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50"
)

# Loop from 400 to 31200 with a step of 400.
# This will generate an array of checkpoint numbers from 400 to 31200.
for ((i = 400; i <= 31200; i += 400)); do
  CHECKPOINT_NO_ARRAY+=("$i")
done

#
# === END Version 2: Checkpoints from training for 50 epochs

# # # #
# Concatenate array elements into a comma-separated string
LANGUAGE_MODEL_LIST=$(
  IFS=,
  echo "${LANGUAGE_MODEL_ARRAY[*]}"
)
CHECKPOINT_NO_LIST=$(
  IFS=,
  echo "${CHECKPOINT_NO_ARRAY[*]}"
)

echo "LANGUAGE_MODEL_ARRAY=${LANGUAGE_MODEL_ARRAY[@]}"
echo "LANGUAGE_MODEL_LIST=${LANGUAGE_MODEL_LIST}"

echo "CHECKPOINT_NO_ARRAY=${CHECKPOINT_NO_ARRAY[@]}"
echo "CHECKPOINT_NO_LIST=${CHECKPOINT_NO_LIST}"

PREFERRED_TORCH_BACKEND="auto"
# PREFERRED_TORCH_BACKEND="cpu"

# Note: Do not set `CUDA_VISIBLE_DEVICES` on HPC cluster,
# as this will lead to the wrong GPU being used.
#
# CUDA_VISIBLE_DEVICES=0

ADDITIONAL_OVERRIDES=""
# ADDITIONAL_OVERRIDES+=" hydra.job.env_set.CUDA_VISIBLE_DEVICES=\"${CUDA_VISIBLE_DEVICES}\""

# ==================================================== #

# TEMPLATE_STRING="A100_40GB"
# TEMPLATE_STRING="RTX6000"
TEMPLATE_STRING="DSML"

# Create the command string in a separate step
COMMAND_ARGS="--multirun"
COMMAND_ARGS+=" language_model=${LANGUAGE_MODEL_LIST}"
COMMAND_ARGS+=" preferred_torch_backend=${PREFERRED_TORCH_BACKEND}"

# Conditionally append the checkpoint number if it is set
if [[ -n "${CHECKPOINT_NO_LIST}" ]]; then
  COMMAND_ARGS+=" language_model.checkpoint_no=${CHECKPOINT_NO_LIST}"
fi

# Append additional overrides if any
COMMAND_ARGS+="${ADDITIONAL_OVERRIDES}"

echo "COMMAND_ARGS=${COMMAND_ARGS}"

# ==================================================== #

hpc run \
  -n "run_inference_pipeline" \
  --template "${TEMPLATE_STRING}" \
  -s "${RELATIVE_PYTHON_SCRIPT_PATH}" \
  -a "${COMMAND_ARGS}"

# ==================================================== #

# Exit with the exit code of the python command
exit $?
