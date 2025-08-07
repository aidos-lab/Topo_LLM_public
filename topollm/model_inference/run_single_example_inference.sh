#!/bin/bash

# This variable is needed to ensure that the script runs on macOS with MPS backend.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# > Very small model
# LANGUAGE_MODEL="gpt2"

# > 1B model
LANGUAGE_MODEL="Llama-3.2-1B"

# LANGUAGE_MODEL="luster-rl-sent"

PROMPTS="['The hotel should be in the', 'I would like to invest in', 'The Eiffel tower is located in', 'Eigenspaces corresponding to distinct eigenvalues are']"

uv run python3 topollm/model_inference/run_inference_pipeline.py \
    --multirun \
    hydra/sweeper="basic" \
    hydra/launcher="basic" \
    preferred_torch_backend="auto" \
    language_model="$LANGUAGE_MODEL" \
    inference.prompts="$PROMPTS" \
    global_seed="1111" \
    inference.include_timestamp_in_filename="True" \
    feature_flags.wandb.use_wandb="False" \
    wandb.project="Topo_LLM_DEBUG"