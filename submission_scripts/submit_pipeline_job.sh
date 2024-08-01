#!/bin/bash

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

echo "TOPO_LLM_REPOSITORY_BASE_PATH=${TOPO_LLM_REPOSITORY_BASE_PATH}"

PYTHON_SCRIPT_NAME="run_pipeline_embeddings_data_prep_local_estimate.py"
RELATIVE_PYTHON_SCRIPT_PATH="topollm/pipeline_scripts/${PYTHON_SCRIPT_NAME}"
ABSOLUTE_PYTHON_SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/${RELATIVE_PYTHON_SCRIPT_PATH}"

# ==================================================== #
# Select the parameters here

# Define arrays for DATA_LIST and DATA_NUMBER_OF_SAMPLES
data_lists=(
    "multiwoz21" 
    "iclr_2024_submissions" 
    "wikitext"
    "one-year-of-tsla-on-reddit"
)
data_samples=(
    "3000" 
    "-1" 
    "-1"
    "3000"
)

language_models=(
    "roberta-base"
    "model-roberta-base_task-MASKED_LM_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5"
    "model-roberta-base_task-MASKED_LM_iclr_2024_submissions-train-5000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5"
    "model-roberta-base_task-MASKED_LM_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5"
    "model-roberta-base_task-MASKED_LM_wikitext-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5"
)


# Note: In the dimension experiments, we usually set `add_prefix_space=False` 
# ADDITIONAL_OVERRIDES+=" tokenizer.add_prefix_space=True"

# LAYER_INDICES_LIST="[-1]"

layer_indices=(
    "[-1]"
    "[-5]"
    "[-9]"
)

EMBEDDINGS_DATA_PREP_NUM_SAMPLES="30000"

ADDITIONAL_OVERRIDES=""

# ==================================================== #

TEMPLATE_STRING="RTX6000"

# Loop over the arrays
for i in "${!data_lists[@]}"; do
    DATA_LIST="${data_lists[$i]}"
    DATA_NUMBER_OF_SAMPLES="${data_samples[$i]}"

    for LANGUAGE_MODEL_LIST in "${language_models[@]}"; do
        for LAYER_INDICES_LIST in "${layer_indices[@]}"; do
            echo "====================================================="
            echo "DATA_LIST=${DATA_LIST}"
            echo "DATA_NUMBER_OF_SAMPLES=${DATA_NUMBER_OF_SAMPLES}"
            echo "LANGUAGE_MODEL_LIST=${LANGUAGE_MODEL_LIST}"
            echo "LAYER_INDICES_LIST=${LAYER_INDICES_LIST}"

            hpc run \
                -n "compute_perplexity_job_submission_manual" \
                -s "${RELATIVE_PYTHON_SCRIPT_PATH}" \
                -a "data="$DATA_LIST" \
                data.number_of_samples="$DATA_NUMBER_OF_SAMPLES" \
                data.split="validation" \
                language_model="$LANGUAGE_MODEL_LIST" \
                embeddings.embedding_extraction.layer_indices=$LAYER_INDICES_LIST \
                embeddings_data_prep.num_samples=$EMBEDDINGS_DATA_PREP_NUM_SAMPLES \
                $ADDITIONAL_OVERRIDES" \
                --template "${TEMPLATE_STRING}" \
                --ncpus 4 \
                --accelerator_model "rtx6000" \
                --queue "CUDA"

            echo "====================================================="
        done
    done
done

# Exit with the return code of the last command
exit $?