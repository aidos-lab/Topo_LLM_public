#!/bin/bash

if [ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]; then
    echo "@@@ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set." >&2
    exit 1
fi

SCRIPT_PATH="${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/DO.run_eval.sh"

echo ">>> Submitting job array ..."

# Note: Currently, the PBS_ARRAY_INDEX is mapped to a seed in the following way:
#
# # Select seed via PBS_ARRAY_INDEX
# SEEDS_SELECTION=(
#     1111 # The first element is the default seed with which the script is run
#     40
#     41
#     42
#     43
#     44
# )
# SEED=${SEEDS_SELECTION[$((PBS_ARRAY_INDEX))]}
#
# E.g., submitting an array with the `-J 1-2` option will select the seeds 40 and 41.

# > Submit job array via PBS with multiple jobs.
# > Note that the SEED in the train.sh script is selected via the PBS_ARRAY_INDEX.
#
qsub -J 1-2 $SCRIPT_PATH

# > Submit with a single job.
# > Note that you cannot use the `-J` option with a single job.
#
# qsub $SCRIPT_PATH

echo ">>> Job array submitted."

echo ">>> Exiting the submission script."
exit 0
