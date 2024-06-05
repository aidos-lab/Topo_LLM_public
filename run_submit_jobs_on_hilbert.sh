#!/bin/bash

# Note: Use `source run_submit_jobs_on_hilbert.sh` to run this script.

# From the instructions at:
# https://gitlab.cs.uni-duesseldorf.de/dsml/user_tools#tools
#
# My bash job
#
# submit_job --job_name my_bash_job --job_script my_bash_script.sh --job_script_args "--arg1 val1 --arg2 val2"
#
# My Python job
#
# submit_job --job_name my_python_job --job_script my_python_script.py --job_script_args "--arg1 val1 --arg2 val2"

echo "TOPO_LLM_REPOSITORY_BASE_PATH=${TOPO_LLM_REPOSITORY_BASE_PATH}"

echo "Calling submit_job ..."

submit_job \
    --job_name "my_bash_job" \
    --job_script "${TOPO_LLM_REPOSITORY_BASE_PATH}/topollm/model_finetuning/run_multiple_finetunings.sh" \
    --memory "64" \
    --ngpus "1" \
    --walltime "08:00:00" \
    --job_script_args ""

echo "Calling submit_job DONE"

# Exit with the exit code of the last command
exit $?