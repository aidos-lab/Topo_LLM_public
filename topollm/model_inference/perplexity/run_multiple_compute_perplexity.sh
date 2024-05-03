#!/bin/bash

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

PYTHON_SCRIPT_NAME="run_compute_perplexity.py"

# ==================================================== #
# Select the parameters here

# DATA_LIST="one-year-of-tsla-on-reddit"
# DATA_LIST="one-year-of-tsla-on-reddit,one-year-of-tsla-on-reddit_validation,multiwoz21_validation,sgd,iclr_2024_submissions,wikitext"
# DATA_LIST="one-year-of-tsla-on-reddit,one-year-of-tsla-on-reddit_validation"
# DATA_LIST="bbc,multiwoz21,sgd,wikitext"
# DATA_LIST="one-year-of-tsla-on-reddit_validation,multiwoz21_validation,sgd,iclr_2024_submissions,wikitext"
DATA_LIST="multiwoz21_validation"


# LANGUAGE_MODEL_LIST="roberta-base"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-one-year-of-tsla-on-reddit_ftm-standard"
# LANGUAGE_MODEL_LIST="roberta-base,roberta-base_finetuned-on-multiwoz21_ftm-lora"
LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-multiwoz21_ftm-standard_overfitted"

ADDITIONAL_OVERRIDES=""
ADDITIONAL_OVERRIDES="data.number_of_samples=3000"
# ADDITIONAL_OVERRIDES+=" language_model.checkpoint_no=400,800,1200,1600,2000,2400,2800"
ADDITIONAL_OVERRIDES+=" language_model.checkpoint_no=400,800,1200,1600,2000,2400,2800,3200,3600,4000,4400,4800,5200,5600,6000,6400,6800,7200,7600,8000,8400,8800,9200,9600,10000,10400,10800,11200,11600,12000,12400,12800,13200,13600,14000,14400,14800,15200,15600"

# ==================================================== #

python3 $PYTHON_SCRIPT_NAME \
    --multirun \
    data=$DATA_LIST \
    language_model=$LANGUAGE_MODEL_LIST \
    hydra.job.env_set.CUDA_VISIBLE_DEVICES=1 \
    $ADDITIONAL_OVERRIDES

# Exit with status code 0
exit 0