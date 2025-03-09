#!/bin/bash

if [ -z "$ZIM_USERNAME" ]; then
    echo ">>> Please set the ZIM_USERNAME environment variable."
    exit 1
fi

echo ">>> Copying file from Hilbert to local machine..."

scp Hilbert-Storage:/gpfs/project/${ZIM_USERNAME}/git-source/Topo_LLM/data/models/setsumbt_checkpoints/rsync_output.txt .

if [ $? -eq 0 ]; then
    echo ">>> File copied successfully."
else
    echo ">>> Error copying file."
fi

exit 0