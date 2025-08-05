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

RELATIVE_SCRIPT_PATH="topollm/pipeline_scripts/"
RELATIVE_SCRIPT_PATH+="run_pipeline_compute_embeddings_and_data_prep_and_local_estimate.py"

ABSOLUTE_SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/${RELATIVE_SCRIPT_PATH}"

usage() {
    echo "💡 Usage: $0 [MODE] [--dry-run]"
    echo "Modes:"
    echo "  compute_embeddings      Run the script to compute embeddings."
    echo "  compute_local_estimates Run the script to compute local estimates."
    echo "Options:"
    echo "  --dry-run               Print the command that would be run, but do not execute it."
    exit 1
}

# --- Parse arguments: allow [MODE] and [--dry-run] in any order ---
MODE=""
dry_run=false

while [[ $# -gt 0 ]]; do
    case $1 in
        compute_embeddings|compute_local_estimates)
            MODE="$1"
            shift
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        *)
            echo "❌ Error: Unknown argument: $1"
            usage
            ;;
    esac
done

# Default mode if not provided
if [[ -z "$MODE" ]]; then
    echo "❌ Error: No mode specified."
    usage
fi

case "$MODE" in
  compute_embeddings)
    LAUNCHER_ARGS=(
        "hydra.launcher.queue=CUDA"
        "hydra.launcher.template=RTX6000"
        "hydra.launcher.ngpus=1"
        "hydra.launcher.memory=64"
        "hydra.launcher.ncpus=2"
        "hydra.launcher.walltime=10:00:00"
    )
    FEATURE_FLAGS_ARGS=(
        "feature_flags.compute_and_store_embeddings.skip_compute_and_store_embeddings_in_pipeline=False"
        "feature_flags.embeddings_data_prep.skip_embeddings_data_prep_in_pipeline=True"
        "feature_flags.analysis.create_plots_in_local_estimates_worker=False"
    )
    ;;
  compute_local_estimates)
    LAUNCHER_ARGS=(
      # Note: 
      # There is a problem with the queue argument "hydra.launcher.queue=DEFAULT",
      # so we do not use it here.
      "hydra.launcher.template=CPU"
      "hydra.launcher.ngpus=0"
      "hydra.launcher.memory=129"
      "hydra.launcher.ncpus=3"
      "hydra.launcher.walltime=04:00:00"
    )
    FEATURE_FLAGS_ARGS=(
        "feature_flags.compute_and_store_embeddings.skip_compute_and_store_embeddings_in_pipeline=True"
        "feature_flags.embeddings_data_prep.skip_embeddings_data_prep_in_pipeline=False"
        "feature_flags.analysis.create_plots_in_local_estimates_worker=True"
    )
    ;;
  *)
    echo "❌ Error: Unknown mode: $MODE"; usage ; exit 1;;
esac
BASE_ARGS=(
    "--multirun"
    "hydra/sweeper=basic"
    "hydra/launcher=hpc_submission"
)


# --- model -----------------------------------------------------------------
LANGUAGE_MODEL_ARGS=(
    # ===== Phi-3.5 and Phi-4 models ===== #
    #
    # "language_model=Phi-3.5-mini-instruct"
    # "language_model='Phi-3.5-mini-instruct-causal_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-random-778_aps-False-mx-512_lora-16-32-o_proj_qkv_proj-0.01-True_5e-05-linear-0.01-5'"
    # "language_model='Phi-3.5-mini-instruct-causal_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-random-778_aps-False-mx-512_lora-16-32-o_proj_qkv_proj-0.01-True_5e-05-linear-0.01-5'"
    # > Two different models:
    # "language_model=Phi-3.5-mini-instruct-causal_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-random-778_aps-False-mx-512_lora-16-32-o_proj_qkv_proj-0.01-True_5e-05-linear-0.01-5,Phi-3.5-mini-instruct-causal_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-random-778_aps-False-mx-512_lora-16-32-o_proj_qkv_proj-0.01-True_5e-05-linear-0.01-5"
    #
    # ===== LLama models ===== #
    #
    # > Smaller (1B and 3B) models:
    # "language_model=Llama-3.2-1B" # <-- (Safetensors: 1.24B parameters)
    # "language_model=Llama-3.2-3B" # <-- (Safetensors: 3.21B parameters)
    # > Two smaller models combined:
    "language_model=Llama-3.2-1B,Llama-3.2-3B"
    #
    # TODO
    #
    # > Checkpoints:
    # "++language_model.checkpoint_no=-1"
    # "++language_model.checkpoint_no=800"
    # "++language_model.checkpoint_no=1200"
    #
)

COMMON_ARGS=(
    # --- memory-friendly settings ---------------------------------------------
    # Note:
    # Increasing to "embeddings.batch_size=8" on the RTX6000-24GB for the "Phi-3.5-mini-instruct" model does NOT work,
    # and leads to a CUDA out of memory error.
    "embeddings.batch_size=4"
    "storage.chunk_size=32"

    "tokenizer.add_prefix_space=False"

    # --- data ------------------------------------------------------------------
    # "data=multiwoz21"
    # "data=wikitext-103-v1"
    "data=multiwoz21,sgd,one-year-of-tsla-on-reddit,wikitext-103-v1,iclr_2024_submissions"
    # > Without wikitext-103-v1:
    # "data=multiwoz21,sgd,one-year-of-tsla-on-reddit,iclr_2024_submissions"
    "data.data_subsampling.split=validation"
    "data.data_subsampling.sampling_mode=random"
    "data.data_subsampling.number_of_samples=10000"
    "data.data_subsampling.sampling_seed=778"

    # --- embeddings & local estimates -----------------------------------------
    "embeddings.embedding_data_handler.mode=regular"
    "embeddings.embedding_extraction.layer_indices=[-1]"
    "embeddings_data_prep.sampling.num_samples=150000"
    "embeddings_data_prep.sampling.sampling_mode=random"
    "embeddings_data_prep.sampling.seed=42"
    "local_estimates=twonn"
    "local_estimates.pointwise.n_neighbors_mode=absolute_size"
    "local_estimates.filtering.deduplication_mode=array_deduplicator"
    "local_estimates.filtering.num_samples=60000"
    "local_estimates.pointwise.absolute_n_neighbors=128"
)

# ---------------------------------------------------------------------------
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo ">> Running script: ${SCRIPT_PATH} ..."

# Build the argument list
ARGS=(
    "${BASE_ARGS[@]}"
    "${LAUNCHER_ARGS[@]}"
    "${LANGUAGE_MODEL_ARGS[@]}"
    "${COMMON_ARGS[@]}"
    "${FEATURE_FLAGS_ARGS[@]}"
    "$@"
)

# Print the argument list
echo "===================================================="
echo ">> Argument list:"
echo "===================================================="
for arg in "${ARGS[@]}"; do
    echo "  $arg"
done
echo "===================================================="

# Run or dry-run the command
if [ "$dry_run" = true ]; then
    echo "💡 [DRY RUN] Would run:"
    echo "uv run python3 $RELATIVE_SCRIPT_PATH ${ARGS[*]}"
else
    uv run python3 "$RELATIVE_SCRIPT_PATH" "${ARGS[@]}"
fi

echo ">> Script completed."
echo "Exiting with last status: $?"
exit $?