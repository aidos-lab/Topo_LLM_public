#!/bin/bash

set -e # Stop script at the first error

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

PYTHON_SCRIPT_NAME="run_single_setup_load_saved_perplexity_and_local_estimates_and_analyse.py"

# ==================================================== #
# Select the parameters here

DATA_LIST="iclr_2024_submissions_test,iclr_2024_submissions_validation"

language_models=(
    "roberta-base"
    # "model-roberta-base_task-MASKED_LM_iclr_2024_submissions-train-5000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5"
    "model-roberta-base_task-MASKED_LM_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5"
    "model-roberta-base_task-MASKED_LM_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5"
    # "model-roberta-base_task-MASKED_LM_wikitext-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5"
)

# LANGUAGE_MODEL_LIST="roberta-base"
LANGUAGE_MODEL_LIST="model-roberta-base_task-MASKED_LM_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5,model-roberta-base_task-MASKED_LM_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5"

LAYER_INDICES_LIST="[-1]"
# LAYER_INDICES_LIST="[-5]"

EMBEDDINGS_DATA_PREP_NUM_SAMPLES="30000"

ADDITIONAL_OVERRIDES=""
# ADDITIONAL_OVERRIDES+="data.number_of_samples=50"
# ADDITIONAL_OVERRIDES+="data.number_of_samples=3000"

# Note: In the dimension experiments, we usually set `add_prefix_space=False` 
# ADDITIONAL_OVERRIDES+=" tokenizer.add_prefix_space=True"


# ==================================================== #

poetry run python3 $PYTHON_SCRIPT_NAME \
    --multirun \
    data=$DATA_LIST \
    language_model=$LANGUAGE_MODEL_LIST \
    embeddings.embedding_extraction.layer_indices=$LAYER_INDICES_LIST \
    embeddings_data_prep.sampling.num_samples=$EMBEDDINGS_DATA_PREP_NUM_SAMPLES \
    +embeddings_data_prep.sampling.sampling_mode="take_first" \
    $ADDITIONAL_OVERRIDES


# Exit with the return code of the last command
exit $?