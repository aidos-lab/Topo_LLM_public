#!/bin/bash

wandb sync --include-offline wandb/offline-*
wandb sync --include-offline wandb/run-*

# Exit with the exit status of the last command
exit $?