#!/bin/bash

set -e # Stop script at the first error


# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

PYTHON_SCRIPT_NAME="run_single_setup_load_saved_perplexity_and_concatenate_into_array.py"

# ==================================================== #
# Select the parameters here

# Define arrays for DATA_LIST and DATA_NUMBER_OF_SAMPLES
data_lists=(
    "multiwoz21" 
    "iclr_2024_submissions" 
    "wikitext"
    # "one-year-of-tsla-on-reddit"
)
data_samples=(
    "10000" 
    "-1" 
    "-1"
    # "3000"
)

language_models=(
    "roberta-base"
    # "model-roberta-base_task-MASKED_LM_iclr_2024_submissions-train-5000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5"
    # "model-roberta-base_task-MASKED_LM_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5"
    # "model-roberta-base_task-MASKED_LM_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5"
    # "model-roberta-base_task-MASKED_LM_wikitext-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5"
)

LAYER_INDICES_LIST="[-1]"
# LAYER_INDICES_LIST="[-5]"

EMBEDDINGS_DATA_PREP_NUM_SAMPLES="30000"

ADDITIONAL_OVERRIDES=""
# ADDITIONAL_OVERRIDES+="data.number_of_samples=50"
# ADDITIONAL_OVERRIDES+="data.number_of_samples=3000"

# Note: In the dimension experiments, we usually set `add_prefix_space=False` 
# ADDITIONAL_OVERRIDES+=" tokenizer.add_prefix_space=True"


# ==================================================== #


# Loop over the arrays
for i in "${!data_lists[@]}"; do
    DATA_LIST="${data_lists[$i]}"
    DATA_NUMBER_OF_SAMPLES="${data_samples[$i]}"

    for LANGUAGE_MODEL_LIST in "${language_models[@]}"; do
        echo "====================================================="
        echo "DATA_LIST=${DATA_LIST}"
        echo "DATA_NUMBER_OF_SAMPLES=${DATA_NUMBER_OF_SAMPLES}"
        echo "LANGUAGE_MODEL_LIST=${LANGUAGE_MODEL_LIST}"

        poetry run python3 $PYTHON_SCRIPT_NAME \
            --multirun \
            data=$DATA_LIST \
            data.number_of_samples="$DATA_NUMBER_OF_SAMPLES" \
            data.split="validation" \
            language_model=$LANGUAGE_MODEL_LIST \
            embeddings.embedding_extraction.layer_indices=$LAYER_INDICES_LIST \
            embeddings_data_prep.sampling.num_samples=$EMBEDDINGS_DATA_PREP_NUM_SAMPLES \
            $ADDITIONAL_OVERRIDES
    done
done

# Exit with the return code of the last command
exit $?