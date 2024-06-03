#!/bin/bash

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

PYTHON_SCRIPT_NAME_COMPUTE_EMBEDDINGS="../compute_embeddings/run_compute_embeddings.py"
PYTHON_SCRIPT_NAME_DATA_PREP="../embeddings_data_prep/run_embeddings_data_prep.py"

# ==================================================== #
# Select the parameters here

# DATA_LIST="one-year-of-tsla-on-reddit_validation,multiwoz21_validation,sgd,iclr_2024_submissions,wikitext,bbc"
# DATA_LIST="one-year-of-tsla-on-reddit_validation,multiwoz21_validation,iclr_2024_submissions,wikitext"
# DATA_LIST="bbc,multiwoz21,sgd,wikitext"
DATA_LIST="multiwoz21_validation,iclr_2024_submissions,wikitext"
# DATA_LIST="multiwoz21_validation"
# DATA_LIST="iclr_2024_submissions"
# DATA_LIST="wikitext"
# DATA_LIST="one-year-of-tsla-on-reddit"
# DATA_LIST="one-year-of-tsla-on-reddit_validation"

LANGUAGE_MODEL_LIST="roberta-base"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-iclr_ftm-standard"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-multiwoz21_ftm-standard"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-multiwoz21_ftm-standard_overfitted"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-one-year-of-tsla-on-reddit_ftm-standard"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-one-year-of-tsla-on-reddit_ftm-standard_overfitted"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-one-year-of-tsla-on-reddit_ftm-standard_freeze-first-6-layers_overfitted"

# LANGUAGE_MODEL_LIST="gpt2-medium"
# LANGUAGE_MODEL_LIST="gpt2-medium_finetuned-on-multiwoz21_ftm-standard"
# LANGUAGE_MODEL_LIST="gpt2-medium_finetuned-on-one-year-of-tsla-on-reddit_ftm-standard"

# LANGUAGE_MODEL_LIST="bert-base-uncased,roberta-base"
# LANGUAGE_MODEL_LIST="roberta-base,roberta-base_finetuned-on-multiwoz21_ftm-lora"
# LANGUAGE_MODEL_LIST="gpt2-medium,gpt2-medium_finetuned-on-multiwoz21_ftm-standard"

# Use the following line to process only the last layer.
# LAYER_INDICES_LIST="[-1]"

LAYER_INDICES_LIST="[-1],[-2]"

# Note that "roberta-base" has 12 layers.
# LAYER_INDICES_LIST="[-1],[-2],[-3],[-4],[-5],[-6],[-7],[-8],[-9],[-10],[-11],[-12]"

# Note that "gpt2-medium" has 24 layers.
# LAYER_INDICES_LIST="[-1],[-3],[-5],[-7],[-9],[-11],[-13],[-15],[-17],[-19],[-21],[-23]"
# LAYER_INDICES_LIST="[-1],[-2],[-3],[-4],[-5],[-6],[-7],[-8],[-9],[-10],[-11],[-12],[-13],[-14],[-15],[-16],[-17],[-18],[-19],[-20],[-21],[-22],[-23],[-24]"

EMBEDDINGS_DATA_PREP_NUM_SAMPLES="30000"
# EMBEDDINGS_DATA_PREP_NUM_SAMPLES="50000"
# EMBEDDINGS_DATA_PREP_NUM_SAMPLES="10000,20000"
# EMBEDDINGS_DATA_PREP_NUM_SAMPLES="5000,10000,15000,20000,25000,30000,50000,75000,100000"

# ADDITIONAL_OVERRIDES=""
# ADDITIONAL_OVERRIDES="data.number_of_samples=1000"
ADDITIONAL_OVERRIDES="data.number_of_samples=3000"
# ADDITIONAL_OVERRIDES+=" language_model.checkpoint_no=400,800,1200"
# ADDITIONAL_OVERRIDES+=" language_model.checkpoint_no=400,800,1200,1600,2000,2400,2800"
# ADDITIONAL_OVERRIDES+=" language_model.checkpoint_no=400,800,1200,1600,2000,2400,2800,3200,3600,4000,4400,4800,5200,5600,6000,6400,6800,7200,7600,8000,8400,8800,9200,9600,10000,10400,10800,11200,11600,12000,12400,12800,13200,13600,14000,14400,14800,15200,15600"
# ADDITIONAL_OVERRIDES+=" language_model.checkpoint_no=400,800,1200,1600,2000,2400,2800,3200,3600,4000,4400,4800,5200,5600,6000,6400,6800,7200,7600,8000,8400,8800,9200,9600,10000,10400,10800,11200,11600,12000,12400,12800,13200,13600,14000,14400,14800,15200,15600,16000,16400,16800,17200,17600,18000,18400,18800,19200,19600,20000,20400,20800,21200,21600,22000,22400,22800,23200,23600,24000,24400,24800,25200,25600,26000,26400,26800,27200,27600,28000,28400,28800,29200,29600,30000,30400,30800,31200"

# Note: Do not overwrite `CUDA_VISIBLE_DEVICES` on HHU hilbert.
# CUDA_VISIBLE_DEVICES=0

# ==================================================== #

echo "Calling python scripts."

python3 $PYTHON_SCRIPT_NAME_COMPUTE_EMBEDDINGS \
    --multirun \
    data=$DATA_LIST \
    language_model=$LANGUAGE_MODEL_LIST \
    embeddings.embedding_extraction.layer_indices=$LAYER_INDICES_LIST \
    hydra.job.env_set.CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    $ADDITIONAL_OVERRIDES

echo "Finished running the first script."

python3 $PYTHON_SCRIPT_NAME_DATA_PREP \
    --multirun \
    data=$DATA_LIST \
    language_model=$LANGUAGE_MODEL_LIST \
    embeddings.embedding_extraction.layer_indices=$LAYER_INDICES_LIST \
    embeddings_data_prep.num_samples=$EMBEDDINGS_DATA_PREP_NUM_SAMPLES \
    $ADDITIONAL_OVERRIDES

echo "Finished running the second script."

# ==================================================== #

echo "Finished running the scripts."
exit 0