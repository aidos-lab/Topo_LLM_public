#!/bin/bash

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

PYTHON_SCRIPT_NAME="twonn.py"

# ==================================================== #
# Select the parameters here

#DATA_LIST="data-wikitext_split-train_ctxt-dataset_entry_samples-1000,data-multiwoz21_split-validation_ctxt-dataset_entry_samples-1000,data-iclr_2024_submissions_split-train_ctxt-dataset_entry_samples-1000"
DATA_LIST="data-multiwoz21_split-validation_ctxt-dataset_entry_samples-1000"
# LANGUAGE_MODEL_LIST="bert-base-uncased,roberta-base"
#LANGUAGE_MODEL_LIST="roberta-base,roberta-base_finetuned-on-multiwoz21_ftm-lora"

#LAYER_INDICES_LIST="[-1],[-2],[-3],[-4],[-5],[-6],[-7],[-8],[-9],[-10],[-11],[-12]"
LAYER_INDICES_LIST="[-1],[-3],[-5],[-7],[-9],[-11],[-13],[-15],[-17],[-19],[-21],[-23]"


#SAMPLE_LIST="10000,20000,30000,50000"
# ==================================================== #

ADDITIONAL_OVERRIDES=""
#ADDITIONAL_OVERRIDES="data.number_of_samples=5000"


python3 $PYTHON_SCRIPT_NAME \
    --multirun \
    data_name=$DATA_LIST \
    layer=$LAYER_INDICES_LIST \


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