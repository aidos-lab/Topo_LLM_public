#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# Note: You need to run this command from castle,
# because we do not have the castle ssh key on the HPC cluster cluster.
scp -rp ruppik@storage.hpc.rz.uni-duesseldorf.de:/gpfs/project/ruppik/models_backup /home/ruppik/models_backup
