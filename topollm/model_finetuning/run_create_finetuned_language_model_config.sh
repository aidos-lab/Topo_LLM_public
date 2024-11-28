#!/bin/bash

python3 run_finetune_language_model_on_huggingface_dataset.py \
    feature_flags.finetuning.skip_finetuning=true \
    feature_flags.wandb.use_wandb=false

# Exit with the exit code of the python command
exit $?