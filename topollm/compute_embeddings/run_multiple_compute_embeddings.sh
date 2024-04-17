#!/bin/bash

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

PYTHON_SCRIPT_NAME="run_compute_embeddings.py"
PYTHON_SCRIPT_NAME_DATA_PREP="data_prep.py"

# ==================================================== #
# Select the parameters here

# DATA_LIST="bbc,multiwoz21,sgd,wikitext"
# DATA_LIST="multiwoz21_validation,iclr_2024_submissions,wikitext"
DATA_LIST="multiwoz21_validation"
# DATA_LIST="iclr_2024_submissions"
# DATA_LIST="wikitext"

# LANGUAGE_MODEL_LIST="roberta-base,roberta-base_finetuned-on-multiwoz21_ftm-lora"
LANGUAGE_MODEL_LIST="gpt2-medium,gpt2-medium_finetuned-on-multiwoz21_ftm-standard"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-multiwoz21_ftm-standard_overfitted"
# LANGUAGE_MODEL_LIST="bert-base-uncased,roberta-base"
# LANGUAGE_MODEL_LIST="roberta-base"
# LANGUAGE_MODEL_LIST="gpt2-medium"
# LANGUAGE_MODEL_LIST="gpt2-medium_finetuned-on-multiwoz21_ftm-standard"
# LANGUAGE_MODEL_LIST="roberta-base_finetuned-on-multiwoz21_ftm-standard_overfitted"

# LAYER_INDICES_LIST="[-1],[-2]"
# LAYER_INDICES_LIST="[-1],[-2],[-3],[-4],[-5],[-6],[-7],[-8],[-9],[-10],[-11],[-12]"
LAYER_INDICES_LIST="[-1],[-3],[-5],[-7],[-9],[-11],[-13],[-15],[-17],[-19],[-21],[-23]"

#DATA_PREP_SAMPLES="10000,20000"
DATA_PREP_SAMPLES="5000,10000,15000,20000,25000,30000,50000"

# ADDITIONAL_OVERRIDES=""
ADDITIONAL_OVERRIDES="data.number_of_samples=1000"

# ==================================================== #


python3 $PYTHON_SCRIPT_NAME \
    --multirun \
    data=$DATA_LIST \
    language_model=$LANGUAGE_MODEL_LIST \
    embeddings.embedding_extraction.layer_indices=$LAYER_INDICES_LIST \
    $ADDITIONAL_OVERRIDES

python3 $PYTHON_SCRIPT_NAME_DATA_PREP \
    --multirun \
    data=$DATA_LIST \
    language_model=$LANGUAGE_MODEL_LIST \
    embeddings.embedding_extraction.layer_indices=$LAYER_INDICES_LIST \
    data_prep_samples=$DATA_PREP_SAMPLES \
    $ADDITIONAL_OVERRIDES