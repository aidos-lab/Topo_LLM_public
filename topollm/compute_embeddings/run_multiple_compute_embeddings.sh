#!/bin/bash

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

PYTHON_SCRIPT_NAME="run_compute_embeddings.py"

DATA_LIST="bbc,multiwoz21,sgd,wikitext"

LANGUAGE_MODEL_LIST="bert-base-uncased,roberta-base"

LAYER_INDICES_LIST="0,1,2,3,4"

ADDITIONAL_OVERRIDES=""
# ADDITIONAL_OVERRIDES="finetuning.max_steps=10"


python3 $PYTHON_SCRIPT_NAME \
    --multirun \
    data=$DATA_LIST \
    language_model=$LANGUAGE_MODEL_LIST \
    embeddings.embedding_extraction.layer_indices.0=$LAYER_INDICES_LIST \
    $ADDITIONAL_OVERRIDES
