#!/bin/bash

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

PYTHON_SCRIPT_NAME="run_finetune_language_model_on_huggingface_dataset.py"

# ==================================================== #
# Select the parameters here

# BASE_MODEL_LIST="gpt2-medium"
BASE_MODEL_LIST="roberta-base"
# BASE_MODEL_LIST="bert-base-uncased,roberta-base"

NUM_TRAIN_EPOCHS="50"

# FINETUNING_DATASETS_LIST="train_and_eval_on_bbc,train_and_eval_on_iclr_2024_submissions,train_and_eval_on_multiwoz21,train_and_eval_on_sgd,train_and_eval_on_wikitext"
# FINETUNING_DATASETS_LIST="train_and_eval_on_iclr_2024_submissions"
# FINETUNING_DATASETS_LIST="train_and_eval_on_multiwoz21"
FINETUNING_DATASETS_LIST="train_and_eval_on_one-year-of-tsla-on-reddit"

# LR_SCHEDULER_TYPE="linear"
LR_SCHEDULER_TYPE="constant"


# TODO(Ben): For "gpt2-medium" using LoRA training, get the following error:
# `ValueError: Target modules {'key', 'value', 'query'} not found in the base model. Please check the target modules and try again.``
#
# PEFT_LIST="lora"
PEFT_LIST="standard"
# PEFT_LIST="standard,lora"

# ADDITIONAL_OVERRIDES=""
# ADDITIONAL_OVERRIDES="finetuning.max_steps=10"

# ==================================================== #

python3 $PYTHON_SCRIPT_NAME \
    --multirun \
    finetuning/base_model@finetuning=$BASE_MODEL_LIST \
    finetuning.num_train_epochs=$NUM_TRAIN_EPOCHS \
    finetuning.lr_scheduler_type=$LR_SCHEDULER_TYPE \
    finetuning.batch_sizes.train=8 \
    finetuning.batch_sizes.eval=8 \
    finetuning/finetuning_datasets=$FINETUNING_DATASETS_LIST \
    finetuning/peft=$PEFT_LIST \
    hydra.job.env_set.CUDA_VISIBLE_DEVICES=0 \
    $ADDITIONAL_OVERRIDES
