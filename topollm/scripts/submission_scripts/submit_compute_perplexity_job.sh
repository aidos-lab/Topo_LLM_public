#!/bin/bash

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

echo "TOPO_LLM_REPOSITORY_BASE_PATH=${TOPO_LLM_REPOSITORY_BASE_PATH}"


PYTHON_SCRIPT_NAME="run_compute_perplexity.py"
RELATIVE_PYTHON_SCRIPT_PATH="topollm/model_inference/perplexity/${PYTHON_SCRIPT_NAME}"
ABSOLUTE_PYTHON_SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/${RELATIVE_PYTHON_SCRIPT_PATH}"


# ==================================================== #
# Select the parameters here

# DATA_LIST="multiwoz21_validation"
# DATA_LIST="sgd_test"
# DATA_LIST="one-year-of-tsla-on-reddit"
# DATA_LIST="one-year-of-tsla-on-reddit_validation"
# DATA_LIST="iclr_2024_submissions"
# DATA_LIST="one-year-of-tsla-on-reddit,one-year-of-tsla-on-reddit_validation,multiwoz21_validation,sgd,iclr_2024_submissions,wikitext"
# DATA_LIST="one-year-of-tsla-on-reddit,one-year-of-tsla-on-reddit_validation"
# DATA_LIST="bbc,multiwoz21,sgd,wikitext"
# DATA_LIST="one-year-of-tsla-on-reddit_validation,multiwoz21_validation,sgd,iclr_2024_submissions,wikitext"
# DATA_LIST="multiwoz21,wikitext,iclr_2024_submissions"
# DATA_LIST="wikitext,iclr_2024_submissions"

DATA_LIST="multiwoz21"
NUMBER_OF_SAMPLES="10000"

# DATA_LIST="iclr_2024_submissions"
# NUMBER_OF_SAMPLES="-1"

# DATA_LIST="wikitext"
# NUMBER_OF_SAMPLES="-1"

# LANGUAGE_MODEL_LIST="roberta-base"
# LANGUAGE_MODEL_LIST="model-roberta-base_task-MASKED_LM_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5"
# LANGUAGE_MODEL_LIST="model-roberta-base_task-MASKED_LM_iclr_2024_submissions-train-5000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5"
# LANGUAGE_MODEL_LIST="model-roberta-base_task-MASKED_LM_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5"
LANGUAGE_MODEL_LIST="model-roberta-base_task-MASKED_LM_wikitext-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5"



ADDITIONAL_OVERRIDES=""

# Note: In the dimension experiments, we usually set `add_prefix_space=False` 
# ADDITIONAL_OVERRIDES+=" tokenizer.add_prefix_space=True"


# ==================================================== #

TEMPLATE_STRING="A100_40GB"

hpc run \
    -n "compute_perplexity_job_submission_manual" \
    -s "${RELATIVE_PYTHON_SCRIPT_PATH}" \
    -a "data="$DATA_LIST" \
    data.number_of_samples="$NUMBER_OF_SAMPLES" \
    data.split="validation" \
    language_model="$LANGUAGE_MODEL_LIST" \
    $ADDITIONAL_OVERRIDES" \
    --template "${TEMPLATE_STRING}"

# Exit with the return code of the last command
exit $?