#!/bin/bash


# # # # # # # # # # # # # # # # # # # # # # # # #
# Default values for arguments
dry_run=false
launcher="basic"   # default launcher
task="do_finetuning"  # default task

# # # # # # # # # # # # # # # # # # # # # # # # #
# Function to print usage
usage() {
    echo "💡 Usage: $0 [--dry-run] [--launcher <basic|hpc_submission>] [--task <do_finetuning|create_language_model_config>] [additional args...]"
    echo "    --dry-run                  : Only print the command, do not execute."
    echo "    --launcher <value>         : Select launcher type ('basic' or 'hpc_submission'). Default: basic."
    echo "    --task <value>             : Select task ('do_finetuning' or 'create_language_model_config'). Default: do_finetuning."
    echo "    [additional args...]       : Any additional arguments will be passed through."
    exit 1
}

# Parse command line options
POSITIONAL_ARGS=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dry-run)
            dry_run=true
            shift
            ;;
        --launcher)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                launcher="$2"
                shift 2
            else
                echo "❌ Error: --launcher requires a value ('basic' or 'hpc_submission')"
                usage
            fi
            ;;
        --task)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                task="$2"
                shift 2
            else
                echo "❌ Error: --task requires a value ('do_finetuning' or 'create_language_model_config')"
                usage
            fi
            ;;
        --help|-h)
            usage
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL_ARGS[@]}"

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

# Log parsed arguments
echo "===================================================="
echo ">> Parsed arguments:"
echo "===================================================="
echo "  dry_run: $dry_run"
echo "  launcher: $launcher"
echo "  task: $task"
echo "  positional args: ${POSITIONAL_ARGS[*]}"
echo "===================================================="

# Launcher selection.
#
# Note: Some LAUNCHER_ARGS might be modified later in the script.
BASE_ARGS=(
    "--multirun" 
    "hydra/sweeper=basic"
)
case "$launcher" in
    basic)
        LAUNCHER_ARGS=(
            "hydra/launcher=basic"
        )
        ;;
    basic_cpu)
        LAUNCHER_ARGS=(
            "hydra/launcher=basic_cpu"
            "preferred_torch_backend=cpu"
        )
        ;;
    hpc_submission)
        LAUNCHER_ARGS=(
            "hydra/launcher=hpc_submission"
            "hydra.launcher.queue=CUDA"
            "hydra.launcher.template=RTX6000"
            "hydra.launcher.ngpus=1"
            "hydra.launcher.memory=64"
            "hydra.launcher.ncpus=2"
            "hydra.launcher.walltime=20:00:00"
        )
        ;;
    *)
        echo "❌ Error: Unknown launcher value '$launcher'."
        usage
        ;;
esac

# Task selection scaffolding
case "$task" in
    do_finetuning)
        FEATURE_FLAGS_ARGS=(
            "feature_flags.finetuning.skip_finetuning=false"
            "feature_flags.wandb.use_wandb=true"
        )
        ;;
    create_language_model_config)
        FEATURE_FLAGS_ARGS=(
            "feature_flags.finetuning.skip_finetuning=true"
            "feature_flags.wandb.use_wandb=false"
        )
        ;;
    *)
        echo "❌ Error: Unknown task value '$task'."
        usage
        ;;
esac

PYTHON_SCRIPT_NAME="run_finetune_language_model_on_huggingface_dataset.py"
RELATIVE_PYTHON_SCRIPT_PATH="topollm/model_finetuning/${PYTHON_SCRIPT_NAME}"

# ==================================================== #
# Select the parameters here
#

# # # # # # # # # # # # # # # # # # # # # # #

# > 1B parameter models:

# BASE_MODEL_LIST="Llama-3.2-1B_for_causal_lm"

# > 3-4B parameter models:

# BASE_MODEL_LIST="Phi-3.5-mini-instruct_for_causal_lm"
BASE_MODEL_LIST="Llama-3.2-3B_for_causal_lm"

# > 8B parameter models:

# BASE_MODEL_LIST="Llama-3.1-8B_for_causal_lm"

# Notes on resource requirements:
#
# - Fine-tuning the "Phi-3.5-mini-instruct_for_causal_lm" model (Safetensors: 3.82B parameters) 
#   with the selected LoRA-configuration (r=16, only targeting attention projection matrices)
#   works on an RTX6000-24GB

# Function to adjust GPU template based on selected language model
adjust_launcher_template_for_model() {
    local model_name="$1"
    if [[ "$BASE_MODEL_LIST" == "Llama-3.1-8B_for_causal_lm" ]]; then
        # Replace template RTX6000 with A100 in LAUNCHER_ARGS
        echo ">> Replacing launcher template with A100"
        for i in "${!LAUNCHER_ARGS[@]}"; do
            if [[ "${LAUNCHER_ARGS[$i]}" == "hydra.launcher.template=RTX6000" ]]; then
                LAUNCHER_ARGS[$i]="hydra.launcher.template=A100_80GB"
            fi
        done
    fi
}
# Adjust the launcher template based on the selected base model
echo ">> Adjusting launcher template for base model: ${BASE_MODEL_LIST} ..."
adjust_launcher_template_for_model "${BASE_MODEL_LIST}"
echo ">> Adjusted LAUNCHER_ARGS:"
echo "${LAUNCHER_ARGS[@]}"

# # # # # # # # # # # # # # # # # # # # # # #

NUM_TRAIN_EPOCHS="5"

SAVE_STEPS="400"
# Note: We only evaluate every 400 steps to avoid a large number of costly evaluation runs.
EVAL_STEPS="400"

# FINETUNING_DATASETS_LIST="train_and_eval_on_multiwoz21_train-samples-small"

FINETUNING_DATASETS_LIST="train_and_eval_on_one-year-of-tsla-on-reddit_train-samples-small,train_and_eval_on_multiwoz21_train-samples-small"
# FINETUNING_DATASETS_LIST="train_and_eval_on_multiwoz21,train_and_eval_on_sgd,train_and_eval_on_wikitext"
# FINETUNING_DATASETS_LIST="train_and_eval_on_bbc,train_and_eval_on_iclr_2024_submissions,train_and_eval_on_multiwoz21,train_and_eval_on_sgd,train_and_eval_on_wikitext"
# FINETUNING_DATASETS_LIST="train_and_eval_on_iclr_2024_submissions"
# FINETUNING_DATASETS_LIST="train_and_eval_on_multiwoz21_10000_samples"
# FINETUNING_DATASETS_LIST="train_and_eval_on_multiwoz21_full"
# FINETUNING_DATASETS_LIST="train_and_eval_on_one-year-of-tsla-on-reddit"

LR_SCHEDULER_TYPE="linear"

PEFT_LIST="lora"
# PEFT_LIST="standard"

LORA_ARGUMENTS_LIST=(
    "finetuning.peft.r=16"
    "finetuning.peft.lora_alpha=32"
    "finetuning.peft.use_rslora=True"
)

# > Target module names for LoRA
# Check which base model you are using:
# - "Phi-3.5-mini-instruct_for_causal_lm": ['o_proj','qkv_proj']
# - "Llama-3.1-8B": ['o_proj','q_proj','k_proj','v_proj']

case "$BASE_MODEL_LIST" in
    "Phi-3.5-mini-instruct_for_causal_lm")
        LORA_ARGUMENTS_LIST+=(
            "finetuning.peft.target_modules=['o_proj','qkv_proj']"
        )
        ;;
    "Llama-3.2-1B_for_causal_lm"|"Llama-3.2-3B_for_causal_lm"|"Llama-3.1-8B_for_causal_lm")
        # Notes:
        # - In addition to the attention layers, we could target the MLP linear layers
        #   ['gate_proj', 'up_proj', 'down_proj'] 
        #   as well.
        LORA_ARGUMENTS_LIST+=(
            "finetuning.peft.target_modules=['o_proj','q_proj','k_proj','v_proj']"
        )
        ;;
    *)
        echo "⚠️ Warning: Unknown base model '$BASE_MODEL_LIST'. Please specify target_modules for LoRA manually."
        echo "⚠️ The script will exit now to avoid potential configuration issues."
        exit 1
        ;;
esac



GRADIENT_MODIFIER_LIST="do_nothing"

COMMON_BATCH_SIZE="8"

BATCH_SIZE_TRAIN="${COMMON_BATCH_SIZE}"
BATCH_SIZE_EVAL="${COMMON_BATCH_SIZE}"

# CUDA_VISIBLE_DEVICES=0

ADDITIONAL_OVERRIDES=""
# ADDITIONAL_OVERRIDES+=" hydra.job.env_set.CUDA_VISIBLE_DEVICES=\"${CUDA_VISIBLE_DEVICES}\""
# Note: You can explicitly set the preferred torch backend to avoid `ValueError: fp16 mixed precision requires a GPU (not 'mps')`. 
# ADDITIONAL_OVERRIDES+=" preferred_torch_backend=cpu"

# Comment out the `finetuning.max_steps` argument for full training.
# Note: Setting this to anything but '-1' will lead to partial training.
#
# ADDITIONAL_OVERRIDES+=" finetuning.max_steps=500"

# ==================================================== #

echo ">>> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Build the argument list
COMMON_ARGS=(
    "finetuning/base_model=${BASE_MODEL_LIST}"
    "finetuning.num_train_epochs=${NUM_TRAIN_EPOCHS}"
    "finetuning.lr_scheduler_type=${LR_SCHEDULER_TYPE}"
    "finetuning.save_steps=${SAVE_STEPS}"
    "finetuning.eval_steps=${EVAL_STEPS}"
    "finetuning.fp16=true"
    "finetuning.batch_sizes.train=${BATCH_SIZE_TRAIN}"
    "finetuning.batch_sizes.eval=${BATCH_SIZE_EVAL}"
    "finetuning/finetuning_datasets=${FINETUNING_DATASETS_LIST}"
    "finetuning/peft=${PEFT_LIST}"
    "${LORA_ARGUMENTS_LIST[@]}"
    "finetuning/gradient_modifier=${GRADIENT_MODIFIER_LIST}"
    "wandb.project=fine_tune_medium_models"
    $ADDITIONAL_OVERRIDES
)

ARGS=(
    "${BASE_ARGS[@]}"
    "${LAUNCHER_ARGS[@]}"
    "${COMMON_ARGS[@]}"
    "${FEATURE_FLAGS_ARGS[@]}"
    "${POSITIONAL_ARGS[@]}"
)

# Print the argument list
echo "===================================================="
echo ">> Argument list:"
echo "===================================================="
for arg in "${ARGS[@]}"; do
    echo "  $arg"
done
echo "===================================================="

echo ">>> Calling script from RELATIVE_PYTHON_SCRIPT_PATH=${RELATIVE_PYTHON_SCRIPT_PATH} ..."

if [ "$dry_run" = true ]; then
    echo "💡 [DRY RUN] Would run:"
    echo "uv run python3 $RELATIVE_PYTHON_SCRIPT_PATH ${ARGS[*]}"
else
    uv run python3 $RELATIVE_PYTHON_SCRIPT_PATH "${ARGS[@]}"
fi

echo ">>> Calling script from RELATIVE_PYTHON_SCRIPT_PATH=${RELATIVE_PYTHON_SCRIPT_PATH} DONE"

# Exit with the exit code of the python command
exit $?
