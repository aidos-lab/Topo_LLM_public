#!/bin/bash

poetry run submit_jobs \
    --task="pipeline"

poetry run submit_jobs \
    --task="perplexity"
