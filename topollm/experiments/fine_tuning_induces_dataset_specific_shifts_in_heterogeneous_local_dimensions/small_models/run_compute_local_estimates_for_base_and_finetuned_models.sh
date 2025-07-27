#!/bin/bash

# # # # # # # # # # # # # # # # # # # # # # # # #
# Check if environment variables are set

if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
    echo "❌ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
    exit 1
fi

# Load environment variables from the .env file
source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# # # # # # # # # # # # # # # # # # # # # # # # #
# Log variables

VARIABLES_TO_LOG_LIST=(
    "TOPO_LLM_REPOSITORY_BASE_PATH" # example: /gpfs/project/$USER/git-source/Topo_LLM
)

for VARIABLE_NAME in "${VARIABLES_TO_LOG_LIST[@]}"; do
    echo "💡 ${VARIABLE_NAME}=${!VARIABLE_NAME}"
done


SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/topollm/pipeline_scripts/run_pipeline_compute_embeddings_and_data_prep_and_local_estimate.py"

export PYTORCH_ENABLE_MPS_FALLBACK=1

# Use the following for HPC cluster submission:
#
# hydra/launcher=hpc_submission \
# hydra.launcher.queue=CUDA \
# hydra.launcher.template=RTX6000 \
# hydra.launcher.memory=32 \
# hydra.launcher.ncpus=2 \
# hydra.launcher.ngpus=1 \
# hydra.launcher.walltime=02:30:00

# # # #
# Compute local estimates for base model:

uv run python3 "${SCRIPT_PATH}" \
    --multirun \
    hydra/sweeper=basic \
    hydra/launcher=basic \
    tokenizer.add_prefix_space=False \
    data=multiwoz21,wikitext-103-v1,one-year-of-tsla-on-reddit \
    data.data_subsampling.split=train,validation,test \
    data.data_subsampling.sampling_mode=random \
    data.data_subsampling.number_of_samples=10000 \
    data.data_subsampling.sampling_seed=778 \
    language_model=roberta-base \
    ++language_model.checkpoint_no=-1 \
    embeddings.embedding_data_handler.mode=regular \
    embeddings.embedding_extraction.layer_indices=[-1] \
    embeddings_data_prep.sampling.num_samples=150000 \
    embeddings_data_prep.sampling.sampling_mode=random \
    embeddings_data_prep.sampling.seed=42 \
    local_estimates=twonn \
    local_estimates.pointwise.n_neighbors_mode=absolute_size \
    local_estimates.filtering.deduplication_mode=array_deduplicator \
    local_estimates.filtering.num_samples=60000 \
    local_estimates.pointwise.absolute_n_neighbors=128 \
    feature_flags.analysis.create_plots_in_local_estimates_worker=False

# # # #
# Compute local estimates for fine-tuned model
# (here as an example for the MultiWOZ-finetuned model)

uv run python3 "${SCRIPT_PATH}" \
    --multirun \
    hydra/sweeper=basic \
    hydra/launcher=basic \
    tokenizer.add_prefix_space=False \
    data=multiwoz21,wikitext-103-v1,one-year-of-tsla-on-reddit \
    data.data_subsampling.split=train,validation,test \
    data.data_subsampling.sampling_mode=random \
    data.data_subsampling.number_of_samples=10000 \
    data.data_subsampling.sampling_seed=778 \
    language_model="roberta-base-masked_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5" \
    ++language_model.checkpoint_no=2800 \
    embeddings.embedding_data_handler.mode=regular \
    embeddings.embedding_extraction.layer_indices=[-1] \
    embeddings_data_prep.sampling.num_samples=150000 \
    embeddings_data_prep.sampling.sampling_mode=random \
    embeddings_data_prep.sampling.seed=42 \
    local_estimates=twonn \
    local_estimates.pointwise.n_neighbors_mode=absolute_size \
    local_estimates.filtering.deduplication_mode=array_deduplicator \
    local_estimates.filtering.num_samples=60000 \
    local_estimates.pointwise.absolute_n_neighbors=128 \
    feature_flags.analysis.create_plots_in_local_estimates_worker=False
