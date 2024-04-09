#!/bin/bash

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

PYTHON_SCRIPT_NAME="run_finetune_masked_language_model_on_huggingface_dataset.py"

# ==================================================== #
# Select the parameters here

# PRETRAINED_MODEL_NAME_OR_PATH_LIST="bert-base-uncased,roberta-base"
# PRETRAINED_MODEL_NAME_OR_PATH_LIST="bert-base-uncased"
PRETRAINED_MODEL_NAME_OR_PATH_LIST="roberta-base"

# FINETUNING_DATASETS_LIST="train_and_eval_on_bbc,train_and_eval_on_iclr_2024_submissions,train_and_eval_on_multiwoz21,train_and_eval_on_sgd,train_and_eval_on_wikitext"
# FINETUNING_DATASETS_LIST="train_and_eval_on_iclr_2024_submissions"
FINETUNING_DATASETS_LIST="train_and_eval_on_multiwoz21"

# ADDITIONAL_OVERRIDES=""
# ADDITIONAL_OVERRIDES="finetuning.max_steps=10"

# ==================================================== #

python3 $PYTHON_SCRIPT_NAME \
    --multirun \
    finetuning.pretrained_model_name_or_path=$PRETRAINED_MODEL_NAME_OR_PATH_LIST \
    finetuning.num_train_epochs=50 \
    finetuning.lr_scheduler_type=constant \
    finetuning/finetuning_datasets=$FINETUNING_DATASETS_LIST \
    finetuning/peft=standard,lora \
    $ADDITIONAL_OVERRIDES
