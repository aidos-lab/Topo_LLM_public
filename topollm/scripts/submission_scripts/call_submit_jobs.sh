#!/bin/bash

DATA_LIST="only_train"

# poetry run submit_jobs \
#     --task="pipeline"
#     --queue="DSML" \
#     --template="DSML" \
#     --data_list=$DATA_LIST

poetry run submit_jobs \
    --task="perplexity" \
    --queue="CUDA" \
    --template="RTX6000" \
    --data_list=$DATA_LIST
