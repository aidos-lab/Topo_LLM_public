#!/bin/bash

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

PYTHON_SCRIPT_NAME="run_load_saved_perplexity_and_concatenate_into_array.py"

# ==================================================== #
# Select the parameters here

DATA_LIST="sgd_test"


# LANGUAGE_MODEL_LIST="roberta-base"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-multiwoz21_ftm-standard_full-dataset"
LANGUAGE_MODEL_LIST="roberta-base,roberta-base_finetuned-on-multiwoz21_ftm-standard_full-dataset"


ADDITIONAL_OVERRIDES=""
# ADDITIONAL_OVERRIDES+="data.number_of_samples=50"
# ADDITIONAL_OVERRIDES+="data.number_of_samples=3000"

# Note: In the dimension experiments, we usually set `add_prefix_space=False` 
ADDITIONAL_OVERRIDES+=" tokenizer.add_prefix_space=True"

CUDA_VISIBLE_DEVICES=0

# ==================================================== #

python3 $PYTHON_SCRIPT_NAME \
    --multirun \
    data=$DATA_LIST \
    language_model=$LANGUAGE_MODEL_LIST \
    hydra.job.env_set.CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    $ADDITIONAL_OVERRIDES

# Exit with status code 0
exit 0