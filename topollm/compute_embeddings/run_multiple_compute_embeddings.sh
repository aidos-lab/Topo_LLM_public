#!/bin/bash

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

PYTHON_SCRIPT_NAME="run_compute_embeddings.py"

# ==================================================== #
# Select the parameters here

# DATA_LIST="bbc,multiwoz21,sgd,wikitext"
DATA_LIST="sgd"

# LANGUAGE_MODEL_LIST="bert-base-uncased,roberta-base"
LANGUAGE_MODEL_LIST="roberta-base,roberta-base_finetuned-on-multiwoz21_ftm-lora"

LAYER_INDICES_LIST="[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11]"

# ==================================================== #

ADDITIONAL_OVERRIDES=""
ADDITIONAL_OVERRIDES="data.number_of_samples=100"

python3 $PYTHON_SCRIPT_NAME \
    --multirun \
    data=$DATA_LIST \
    language_model=$LANGUAGE_MODEL_LIST \
    embeddings.embedding_extraction.layer_indices=$LAYER_INDICES_LIST \
    $ADDITIONAL_OVERRIDES

# Alternative way to specify the list of layer indices override:
#
# Define the list of layer indices as a string:
# > LAYER_INDICES_LIST="0,1,2,3,4"
# The in the run command, add the following line:
# > embeddings.embedding_extraction.layer_indices=$LAYER_INDICES_LIST
#
# Note that the syntax "layer_indices.0" is used to specify the first element
# of the list "layer_indices" in the configuration file.
# https://stackoverflow.com/questions/77940195/overriding-list-entry-in-hydra-from-another-yaml