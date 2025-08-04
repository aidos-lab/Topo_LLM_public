#!/bin/bash

PROMPTS="['The hotel should be in the', 'I would like to invest in', 'The Eiffel tower is located in', 'Eigenspaces corresponding to distinct eigenvalues are']"

uv run python3 topollm/model_inference/run_inference_pipeline.py \
    --multirun \
    hydra/sweeper="basic" \
    hydra/launcher="basic" \
    preferred_torch_backend="auto" \
    language_model="luster-rl-sent" \
    inference.prompts="$PROMPTS" \
    global_seed="1111" \
    inference.include_timestamp_in_filename="True" \
    feature_flags.wandb.use_wandb="False" \
    wandb.project="Topo_LLM_DEBUG"