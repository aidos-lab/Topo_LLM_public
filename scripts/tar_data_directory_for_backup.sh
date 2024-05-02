#!/bin/bash

DATA_DIR="data"
BACKUP_DIR="${DATA_DIR}/backup"

INPUT_FOLDER="${DATA_DIR}/embeddings"
OUTPUT_PATH="${BACKUP_DIR}/2024-05-01_BACKUP_embeddings.tar.gz"

echo "INPUT_FOLDER: ${INPUT_FOLDER}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"

# Create the output directory if it does not exist
mkdir -p $(dirname $OUTPUT_PATH)

# https://superuser.com/questions/168749/is-there-a-way-to-see-any-tar-progress-per-file
tar cf - $INPUT_FOLDER -P | pv -s $(du -sb $INPUT_FOLDER | awk '{print $1}') | gzip > $OUTPUT_PATH