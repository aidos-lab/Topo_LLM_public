#!/bin/bash -li
#PBS -l select=1:ncpus=2:mem=32gb:ngpus=1
#PBS -l walltime=59:59:59
#PBS -A "ANONYMIZED_ACCOUNT"
#PBS -q "ANONYMIZED_QUEUE"
#PBS -r y
#PBS -e /gpfs/project/ANONYMIZED_ACCOUNT/job_logs
#PBS -o /gpfs/project/ANONYMIZED_ACCOUNT/job_logs
#PBS -N train_contextbert_ertod_main_0

# ======================================================================================================= #
# Notes:
# - PBS cannot submit non-rerunable Array Jobs, thus we need to set the rerunable flag to "-r y" in the PBS header.
# - For a shorter runtime (5 epochs), the following walltime should be enough: #PBS -l walltime=12:00:00
#
# >>> Alternatives:
# - Use the CUDA queue:
# > PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:accelerator_model=gtx1080ti
# > PBS -A "ANONYMIZED_ACCOUNT"
# > PBS -q 'CUDA'
#
# ======================================================================================================= #

# # # # # # # # # # # # # # # #
# Load environment

# Note: This is where you can add environment specific setup.

# Check that environment variables are set
if [ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]; then
    echo "@@@ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set." >&2
    exit 1
fi
if [ -z "${CONVLAB3_REPOSITORY_BASE_PATH}" ]; then
    echo "@@@ Error: CONVLAB3_REPOSITORY_BASE_PATH is not set." >&2
    exit 1
fi

# If $PBS_ARRAY_INDEX is not set, set it to 0
if [ -z "${PBS_ARRAY_INDEX}" ]; then
    echo "@@@ Warning: PBS_ARRAY_INDEX is not set. Setting it to 0." >&2
    PBS_ARRAY_INDEX=0
fi

# Move to project folder
cd "${CONVLAB3_REPOSITORY_BASE_PATH}"

# # # # # # # # # # # # # # # #
# START: Parameters

DEBUG_TRUNCATION_SIZE="-1"
# DEBUG_TRUNCATION_SIZE="60"

USE_CONTEXT="False"

# EPOCHS="5"
EPOCHS="50"

# Select seed via PBS_ARRAY_INDEX
SEEDS=(
    49
    50
    51
    52
    53
)
SEED=${SEEDS[$((PBS_ARRAY_INDEX))]}

# END: Parameters
# # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # #
# Build paths and flags

SCRIPT_PATH="${CONVLAB3_REPOSITORY_BASE_PATH}/convlab/dst/emodst/modeling/train_contextbert_ertod.py"

# Set the flag for using context in training.
if [[ "${USE_CONTEXT}" == "True" ]]; then
    USE_CONTEXT_FLAG="--use_context"
elif [[ "${USE_CONTEXT}" == "False" ]]; then
    USE_CONTEXT_FLAG=""
else
    echo "@@@ Error: USE_CONTEXT must be either 'True' or 'False'" >&2
    exit 1
fi

OUTPUT_ROOT_DIR="${TOPO_LLM_REPOSITORY_BASE_PATH}/"
OUTPUT_ROOT_DIR+="data/models/EmoLoop/output_dir/"
OUTPUT_ROOT_DIR+="debug=${DEBUG_TRUNCATION_SIZE}/"
OUTPUT_ROOT_DIR+="use_context=${USE_CONTEXT}/"
OUTPUT_ROOT_DIR+="ep=${EPOCHS}/"
OUTPUT_ROOT_DIR+="seed=${SEED}"

VARIABLES_TO_LOG=(
    "PBS_ARRAY_INDEX"
    "TOPO_LLM_REPOSITORY_BASE_PATH"
    "CONVLAB3_REPOSITORY_BASE_PATH"
    "USE_CONTEXT"
    "USE_CONTEXT_FLAG"
    "EPOCHS"
    "SEED"
    "DEBUG_TRUNCATION_SIZE"
    "SCRIPT_PATH"
    "OUTPUT_ROOT_DIR"
)

echo ""
echo ">>> -------------------------------------"
echo ">>> Environment variables:"
echo ">>> -------------------------------------"
for var in "${VARIABLES_TO_LOG[@]}"; do
    echo ">>> $var=${!var}"
done
echo ">>> -------------------------------------"
echo ""

# # # # # # # # # # # # # # # #
# Run the training script

echo ">>> Calling python training script ..."

# Notes:
# - Do not add quotation marks around $USE_CONTEXT_FLAG in the command below,
#   since this will be incorrectly interpreted if USE_CONTEXT_FLAG is the empty string.

poetry run python3 "${SCRIPT_PATH}" \
    --exp_id "${OUTPUT_ROOT_DIR}" \
    --data_dir "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/EmoLoop/required_files/dst/data" \
    --pretrained_model_dir "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/EmoLoop/required_files/dst/bert-base-uncased" \
    --data_dfs_save_dir "${OUTPUT_ROOT_DIR}/data_dfs" \
    --extracted_bert_components_save_dir "${OUTPUT_ROOT_DIR}/extracted_bert_components" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --do_train \
    $USE_CONTEXT_FLAG \
    --emotion \
    --debug_truncation_size "${DEBUG_TRUNCATION_SIZE}"

echo ">>> Calling python training script DONE"

# ======================================================================================================= #
#
# Note: We do not add the following options here, since we wand to train an emotion classification model
# with the most basic settings.
#
# --augment fearful apologetic abusive excited \
# --augment_src to-inferred \
# --dialog_state \
# --valence \
# --elicitor \
# --conduct \
# --distance_loss
#
# ======================================================================================================= #

echo ">>> Bash script DONE"
exit 0
