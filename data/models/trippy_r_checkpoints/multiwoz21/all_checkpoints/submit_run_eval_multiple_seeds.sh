#!/bin/bash

if [ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]; then
    echo "@@@ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set." >&2
    exit 1
fi

echo ">>> Submitting job array ..."

# > Submit job array via PBS with 4 jobs.
# > Note that the SEED in the train.sh script is selected via the PBS_ARRAY_INDEX.
# qsub -J 1-4 "${TOPO_LLM_REPOSITORY_BASE_PATH}/convlab/dst/emodst/modeling/train.sh"

# > Submit with a single job.
qsub "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/DO.run_eval.sh"

echo ">>> Job array submitted."

echo ">>> Exiting the submission script."
exit 0