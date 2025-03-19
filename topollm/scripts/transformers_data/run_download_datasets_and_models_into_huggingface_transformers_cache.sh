
#!/bin/bash

# # # # # # # # # # # # # # # # # # # # # # # # #
# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo ">>> Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  exit 1
fi

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/topollm/scripts/transformers_data/download_datasets_and_models_into_huggingface_transformers_cache.py"

poetry run python3 "${SCRIPT_PATH}" \
    --dataset_names \
    "wikitext" \
    --model_names \
    "roberta-base" \
    "facebook/bart-base"


echo "Exiting script."
exit 0