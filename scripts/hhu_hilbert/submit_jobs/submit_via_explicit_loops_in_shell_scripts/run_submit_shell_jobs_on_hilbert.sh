#!/bin/bash

# Note: Use `source run_submit_shell_jobs_on_hilbert.sh` to run this script.

# From the instructions at:
# https://gitlab.cs.uni-duesseldorf.de/dsml/user_tools#tools
#
# My bash job
#
# submit_job --job_name my_bash_job --job_script my_bash_script.sh --job_script_args "--arg1 val1 --arg2 val2"

echo "TOPO_LLM_REPOSITORY_BASE_PATH=${TOPO_LLM_REPOSITORY_BASE_PATH}"


ACCELERATOR_MODEL="teslat4"
# ACCELERATOR_MODEL="rtx6000"

echo "Calling submit_job ..."

submit_job \
    --job_name "my_bash_job" \
    --job_script "${TOPO_LLM_REPOSITORY_BASE_PATH}/topollm/model_finetuning/run_multiple_finetunings.sh" \
    --ncpus "2" \
    --memory "32" \
    --ngpus "1" \
    --accelerator_model "${ACCELERATOR_MODEL}" \
    --queue "CUDA" \
    --walltime "08:00:00" \
    --job_script_args ""

echo "Calling submit_job DONE"

# Note: Do not add an `exit` command here, since we will source this script, and we want to keep the shell open.