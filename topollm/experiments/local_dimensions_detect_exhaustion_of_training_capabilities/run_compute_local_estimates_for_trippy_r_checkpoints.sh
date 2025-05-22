#!/bin/bash

if [ -z "${TOPO_LLM_REPOSITORY_BASE_PATH:-}" ]; then
    echo "@@@ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set." >&2
    exit 1
fi
echo "TOPO_LLM_REPOSITORY_BASE_PATH=${TOPO_LLM_REPOSITORY_BASE_PATH}"

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
# Notes:
# - The splits for Trippy-R are called: train,dev,test

# # # #
# Compute local estimates for the Trippy-R model checkpoints

uv run python3 "${SCRIPT_PATH}" \
    --multirun \
    hydra/sweeper=basic \
    hydra/launcher=basic \
    tokenizer.add_prefix_space=False \
    data=trippy_r_dataloaders_processed \
    data.data_subsampling.split=train,dev,test \
    data.data_subsampling.sampling_mode=random \
    data.data_subsampling.number_of_samples=7000 \
    data.data_subsampling.sampling_seed=778 \
    language_model="roberta-base-trippy_r_multiwoz21_short_runs" \
    language_model.checkpoint_no=1775,3550,5325,7100,8875,10650,12425,14200,15975,17750,19525,21300,23075,24850,26625,28400,30175,31950,33725,35500 \
    language_model.seed=40,41,42,43,44 \
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
# Compute local estimates for base model

uv run python3 "${SCRIPT_PATH}" \
    --multirun \
    hydra/sweeper=basic \
    hydra/launcher=basic \
    tokenizer.add_prefix_space=False \
    data=trippy_r_dataloaders_processed \
    data.data_subsampling.split=train,dev,test \
    data.data_subsampling.sampling_mode=random \
    data.data_subsampling.number_of_samples=7000 \
    data.data_subsampling.sampling_seed=778 \
    language_model="roberta-base" \
    ++language_model.checkpoint_no=-1 \
    ++language_model.seed=-1 \
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
