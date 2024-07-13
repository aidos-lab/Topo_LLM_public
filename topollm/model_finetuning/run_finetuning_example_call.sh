#!/bin/bash

echo "TOPO_LLM_REPOSITORY_BASE_PATH=${TOPO_LLM_REPOSITORY_BASE_PATH}"

PYTHON_SCRIPT_NAME="run_finetune_language_model_on_huggingface_dataset.py"
PYTHON_SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/topollm/model_finetuning/${PYTHON_SCRIPT_NAME}"

poetry run python3 \
    $PYTHON_SCRIPT_NAME \
    finetuning=finetuning_for_token_classification \
    finetuning.compute_metrics_mode=NONE \
    finetuning.finetuning_datasets.train_dataset.number_of_samples=3000 \
    finetuning.finetuning_datasets.eval_dataset.number_of_samples=200 \
    finetuning.eval_steps=100 \
    +finetuning.trainer_modifier.mode=ADD_WANDB_PREDICTION_PROGRESS_CALLBACK \
    +finetuning.trainer_modifier.frequency=100 \
    feature_flags.wandb.use_wandb=true \
    wandb.project=Topo_LLM_finetuning_for_token_classification_debug
    
    

# Exit with the exit code of the python command
exit $?