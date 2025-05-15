#!/bin/bash

SCRIPT_PATH="topollm/task_performance_analysis/plotting/run_create_distribution_plots_of_local_estimates.py"

add_legend="True"
# add_legend="True,False"

uv run $SCRIPT_PATH \
    --multirun \
    hydra/sweeper=basic \
    hydra/launcher=basic \
    tokenizer.add_prefix_space=False \
    analysis/task_performance_analysis/plotting=ertod \
    analysis.task_performance_analysis.plotting.publication_ready=True \
    analysis.task_performance_analysis.plotting.add_legend=$add_legend \
    analysis.task_performance_analysis.plotting.maximum_x_value=7 \
    feature_flags.task_performance_analysis.plotting_create_distribution_plots_over_model_checkpoints=False \
    feature_flags.task_performance_analysis.plotting_create_mean_plots_over_model_checkpoints_with_different_seeds=True \
    feature_flags.task_performance_analysis.plotting_create_distribution_plots_over_model_layers=False
