#!/bin/bash

# Remark: All paths in this file are relative to the folders set in the environment variables
export TERM_EXTRACTION_BASE_PATH="$HOME/git-source/ConvLab3/convlab/term_extraction"
export TDA_BASE_PATH="$HOME/git-source/ConvLab3/convlab/tda/tda_contextual_embeddings"


# -------------------------

python -u finetune_masked_language_model_with_transformers_on_dialogue_data.py \
    --max_length "256" \
    --train_batch_size "4" \
    --eval_batch_size "2" \
    --data_set_desc_list multiwoz21
    # --debug_index "500"

# -------------------------


