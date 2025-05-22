#!/bin/bash

if [ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]; then
    echo "@@@ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set." >&2
    exit 1
fi
echo "TOPO_LLM_REPOSITORY_BASE_PATH=${TOPO_LLM_REPOSITORY_BASE_PATH}"

SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/topollm/task_performance_analysis/plotting/run_create_distribution_plots_of_local_estimates.py"

uv run python3 "${SCRIPT_PATH}" \
    --multirun \
    hydra/sweeper=basic \
    hydra/launcher=basic \
    analysis/task_performance_analysis/plotting=ertod \
    analysis.task_performance_analysis.plotting.publication_ready=True \
    analysis.task_performance_analysis.plotting.add_legend=True,False \
    analysis.task_performance_analysis.plotting.maximum_x_value=null \
    feature_flags.task_performance_analysis.plotting_create_distribution_plots_over_model_checkpoints=False \
    feature_flags.task_performance_analysis.plotting_create_mean_plots_over_model_checkpoints_with_different_seeds=True \
    feature_flags.task_performance_analysis.plotting_create_distribution_plots_over_model_layers=False
