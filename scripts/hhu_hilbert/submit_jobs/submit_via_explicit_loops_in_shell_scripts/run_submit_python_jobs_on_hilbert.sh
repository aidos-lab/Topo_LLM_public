#!/bin/bash

# Note: Use `source run_submit_python_jobs_on_hilbert.sh` to run this script.

# From the instructions at:
# https://gitlab.cs.uni-duesseldorf.de/dsml/user_tools#tools
#
# My Python job
#
# submit_job --job_name my_python_job --job_script my_python_script.py --job_script_args "--arg1 val1 --arg2 val2"

echo "TOPO_LLM_REPOSITORY_BASE_PATH=${TOPO_LLM_REPOSITORY_BASE_PATH}"

FINETUNING_PYTHON_SCRIPT_NAME="run_finetune_language_model_on_huggingface_dataset.py"
FINETUNING_PYTHON_SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/topollm/model_finetuning/${FINETUNING_PYTHON_SCRIPT_NAME}"

# # # # # # # # #
# Machine setup

# ACCELERATOR_MODEL="teslat4"
ACCELERATOR_MODEL="rtx6000"

# # # # # # # # #
# Script setup

BASE_MODEL_LIST=(
    # "bert-base-uncased"
    "roberta-base"
)

NUM_TRAIN_EPOCHS="5"

COMMON_BATCH_SIZE="16"

BATCH_SIZE_TRAIN="${COMMON_BATCH_SIZE}"
BATCH_SIZE_EVAL="${COMMON_BATCH_SIZE}"

SAVE_STEPS="400"
EVAL_STEPS="100"

FINETUNING_DATASETS_LIST=(
    "train_and_eval_on_multiwoz21_10000_samples"
    "train_and_eval_on_one-year-of-tsla-on-reddit"
)

LR_SCHEDULER_TYPE="linear"

PEFT_LIST=(
    "standard"
    "lora"
)

GRADIENT_MODIFIER_LIST=(
    "do_nothing"
    "freeze_first_layers_bert-style-models"
)

LORA_R_LIST=(
    "16"
    # "32"
    # "64"
)

# # # # # # # # # # # # # # # #
# Run loop

for BASE_MODEL in "${BASE_MODEL_LIST[@]}"; do
    for FINETUNING_DATASET in "${FINETUNING_DATASETS_LIST[@]}"; do
        for PEFT in "${PEFT_LIST[@]}"; do
            for GRADIENT_MODIFIER in "${GRADIENT_MODIFIER_LIST[@]}"; do
                for LORA_R in "${LORA_R_LIST[@]}"; do
                    echo "BASE_MODEL=${BASE_MODEL}"
                    echo "FINETUNING_DATASET=${FINETUNING_DATASET}"
                    echo "PEFT=${PEFT}"
                    echo "GRADIENT_MODIFIER=${GRADIENT_MODIFIER}"
                    echo "LORA_R=${LORA_R}"
                
                    # JOB_SCRIPT_ARGS="\""
                    JOB_SCRIPT_ARGS+="--multirun"
                    JOB_SCRIPT_ARGS+=" finetuning/base_model@finetuning=\"${BASE_MODEL}\""
                    JOB_SCRIPT_ARGS+=" finetuning.num_train_epochs=\"${NUM_TRAIN_EPOCHS}\""
                    JOB_SCRIPT_ARGS+=" finetuning.lr_scheduler_type=\"${LR_SCHEDULER_TYPE}\""
                    JOB_SCRIPT_ARGS+=" finetuning.batch_sizes.train=\"${BATCH_SIZE_TRAIN}\""
                    JOB_SCRIPT_ARGS+=" finetuning.batch_sizes.eval=\"${BATCH_SIZE_EVAL}\""
                    JOB_SCRIPT_ARGS+=" finetuning.save_steps=\"${SAVE_STEPS}\""
                    JOB_SCRIPT_ARGS+=" finetuning.eval_steps=\"${EVAL_STEPS}\""
                    JOB_SCRIPT_ARGS+=" finetuning.fp16=\"true\""
                    JOB_SCRIPT_ARGS+=" finetuning/finetuning_datasets=\"${FINETUNING_DATASET}\""
                    JOB_SCRIPT_ARGS+=" finetuning/peft=\"${PEFT}\""
                    JOB_SCRIPT_ARGS+=" finetuning/gradient_modifier=\"${GRADIENT_MODIFIER}\""
                    JOB_SCRIPT_ARGS+=" ++finetuning.peft.r=\"${LORA_R}\""
                    # JOB_SCRIPT_ARGS+="\""

                    echo "JOB_SCRIPT_ARGS=${JOB_SCRIPT_ARGS}"

                    # # # # # # # # # # # # # # # #
                    echo "Calling submit_job ..."

                    submit_job \
                        --job_name "my_python_job" \
                        --job_script "${FINETUNING_PYTHON_SCRIPT_PATH}" \
                        --ncpus "2" \
                        --memory "32" \
                        --ngpus "1" \
                        --accelerator_model "${ACCELERATOR_MODEL}" \
                        --queue "CUDA" \
                        --walltime "08:00:00" \
                        --job_script_args "${JOB_SCRIPT_ARGS}"

                    echo "Calling submit_job DONE"
                    # # # # # # # # # # # # # # # #
                done
            done
        done
    done
done

# Note: Do not add an `exit` command here, since we will source this script, and we want to keep the shell open.