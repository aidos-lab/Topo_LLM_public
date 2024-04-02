#!/bin/bash

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

PYTHON_SCRIPT_NAME="run_finetune_masked_language_model_on_huggingface_dataset.py"

python3 $PYTHON_SCRIPT_NAME \
    --multirun \
    finetuning/finetuning_datasets=train_and_eval_on_wikitext,train_and_eval_on_sgd \
    finetuning/peft=standard,lora \
    finetuning.max_steps=10
