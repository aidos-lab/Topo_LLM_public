#!/bin/bash

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

PYTHON_SCRIPT_NAME="run_compute_perplexity.py"

# ==================================================== #
# Select the parameters here

# DATA_LIST="multiwoz21_validation"
# DATA_LIST="one-year-of-tsla-on-reddit"
DATA_LIST="one-year-of-tsla-on-reddit_validation"
# DATA_LIST="one-year-of-tsla-on-reddit,one-year-of-tsla-on-reddit_validation,multiwoz21_validation,sgd,iclr_2024_submissions,wikitext"
# DATA_LIST="one-year-of-tsla-on-reddit,one-year-of-tsla-on-reddit_validation"
# DATA_LIST="bbc,multiwoz21,sgd,wikitext"
# DATA_LIST="one-year-of-tsla-on-reddit_validation,multiwoz21_validation,sgd,iclr_2024_submissions,wikitext"


# LANGUAGE_MODEL_LIST="roberta-base"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-multiwoz21_ftm-standard"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-multiwoz21_ftm-standard_overfitted"
# LANGUAGE_MODEL_LIST="roberta-base,roberta-base_finetuned-on-multiwoz21_ftm-lora"

# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-one-year-of-tsla-on-reddit_ftm-standard"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-one-year-of-tsla-on-reddit_ftm-standard_overfitted"
LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-one-year-of-tsla-on-reddit_ftm-standard_freeze-first-6-layers_overfitted"

ADDITIONAL_OVERRIDES=""
ADDITIONAL_OVERRIDES="data.number_of_samples=3000"

# The following line are checkpoints from 400 to 2800 (for ep-5 and batch size 8)
# ADDITIONAL_OVERRIDES+=" language_model.checkpoint_no=400,800,1200,1600,2000,2400,2800"

# The following line are checkpoints from 400 to 15600 (for ep-50 and batch size 16)
# ADDITIONAL_OVERRIDES+=" language_model.checkpoint_no=400,800,1200,1600,2000,2400,2800,3200,3600,4000,4400,4800,5200,5600,6000,6400,6800,7200,7600,8000,8400,8800,9200,9600,10000,10400,10800,11200,11600,12000,12400,12800,13200,13600,14000,14400,14800,15200,15600"

# The following line are checkpoints from 400 to 31200 (for ep-50 and batch size 8)
ADDITIONAL_OVERRIDES+=" language_model.checkpoint_no=400,800,1200,1600,2000,2400,2800,3200,3600,4000,4400,4800,5200,5600,6000,6400,6800,7200,7600,8000,8400,8800,9200,9600,10000,10400,10800,11200,11600,12000,12400,12800,13200,13600,14000,14400,14800,15200,15600,16000,16400,16800,17200,17600,18000,18400,18800,19200,19600,20000,20400,20800,21200,21600,22000,22400,22800,23200,23600,24000,24400,24800,25200,25600,26000,26400,26800,27200,27600,28000,28400,28800,29200,29600,30000,30400,30800,31200"

CUDA_VISIBLE_DEVICES=1

# ==================================================== #

python3 $PYTHON_SCRIPT_NAME \
    --multirun \
    data=$DATA_LIST \
    language_model=$LANGUAGE_MODEL_LIST \
    hydra.job.env_set.CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    $ADDITIONAL_OVERRIDES

# Exit with status code 0
exit 0