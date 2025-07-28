#!/bin/bash

# See the following for details on how to use Hydra multirun:
# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

echo "TOPO_LLM_REPOSITORY_BASE_PATH=${TOPO_LLM_REPOSITORY_BASE_PATH}"

PYTHON_SCRIPT_NAME="run_finetune_language_model_on_huggingface_dataset.py"
PYTHON_SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/topollm/model_finetuning/${PYTHON_SCRIPT_NAME}"

# ==================================================== #
# Select the parameters here

# BASE_MODEL_LIST="gpt2-medium_for_causal_lm"
BASE_MODEL_LIST="roberta-base_for_masked_lm"

NUM_TRAIN_EPOCHS="5"
# NUM_TRAIN_EPOCHS="50"

SAVE_STEPS="400"
EVAL_STEPS="100"

FINETUNING_DATASETS_LIST="train_and_eval_on_one-year-of-tsla-on-reddit_train-samples-small,train_and_eval_on_multiwoz21_train-samples-small"
# FINETUNING_DATASETS_LIST="train_and_eval_on_multiwoz21,train_and_eval_on_sgd,train_and_eval_on_wikitext"
# FINETUNING_DATASETS_LIST="train_and_eval_on_bbc,train_and_eval_on_iclr_2024_submissions,train_and_eval_on_multiwoz21,train_and_eval_on_sgd,train_and_eval_on_wikitext"
# FINETUNING_DATASETS_LIST="train_and_eval_on_iclr_2024_submissions"
# FINETUNING_DATASETS_LIST="train_and_eval_on_multiwoz21_10000_samples"
# FINETUNING_DATASETS_LIST="train_and_eval_on_multiwoz21_full"
# FINETUNING_DATASETS_LIST="train_and_eval_on_one-year-of-tsla-on-reddit"

LR_SCHEDULER_TYPE="linear"
# LR_SCHEDULER_TYPE="constant"

# PEFT_LIST="lora"
PEFT_LIST="standard"
# PEFT_LIST="standard,lora"

GRADIENT_MODIFIER_LIST="do_nothing"
# GRADIENT_MODIFIER_LIST="freeze_first_layers_bert-style-models"
# GRADIENT_MODIFIER_LIST="do_nothing,freeze_first_layers_bert-style-models"

COMMON_BATCH_SIZE="8"

BATCH_SIZE_TRAIN="${COMMON_BATCH_SIZE}"
BATCH_SIZE_EVAL="${COMMON_BATCH_SIZE}"

# CUDA_VISIBLE_DEVICES=0

ADDITIONAL_OVERRIDES=""
# ADDITIONAL_OVERRIDES+=" hydra.job.env_set.CUDA_VISIBLE_DEVICES=\"${CUDA_VISIBLE_DEVICES}\""
# ADDITIONAL_OVERRIDES+=" ++finetuning.peft.r=16"
# ADDITIONAL_OVERRIDES+=" ++finetuning.peft.lora_alpha=32"
# ADDITIONAL_OVERRIDES+=" ++finetuning.peft.use_rslora=True"

# Comment out the `finetuning.max_steps` argument for full training.
# Note: Setting this to anything but '-1' will lead to partial training.
#
# ADDITIONAL_OVERRIDES+=" finetuning.max_steps=500"

# ==================================================== #

echo "Calling script from PYTHON_SCRIPT_PATH=${PYTHON_SCRIPT_PATH} ..."

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

uv run python3 $PYTHON_SCRIPT_PATH \
    --multirun \
    finetuning/base_model="${BASE_MODEL_LIST}" \
    finetuning.num_train_epochs="${NUM_TRAIN_EPOCHS}" \
    finetuning.lr_scheduler_type="${LR_SCHEDULER_TYPE}" \
    finetuning.save_steps="${SAVE_STEPS}" \
    finetuning.eval_steps="${EVAL_STEPS}" \
    finetuning.fp16="true" \
    finetuning.batch_sizes.train="${BATCH_SIZE_TRAIN}" \
    finetuning.batch_sizes.eval="${BATCH_SIZE_EVAL}" \
    finetuning/finetuning_datasets="${FINETUNING_DATASETS_LIST}" \
    finetuning/peft="${PEFT_LIST}" \
    finetuning/gradient_modifier="${GRADIENT_MODIFIER_LIST}" \
    $ADDITIONAL_OVERRIDES

echo "Calling script from PYTHON_SCRIPT_PATH=$PYTHON_SCRIPT_PATH DONE"

# Exit with the exit code of the python command
exit $?
