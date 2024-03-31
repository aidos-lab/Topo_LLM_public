#!/bin/bash

# This script sets up the environment variables for the Topo_LLM repository.
#
# Note: This script only needs to be run once. It sets up the environment
# variables in the .bashrc and .zshrc files in the home directory.


echo "Setting up environment variables for Topo_LLM"

# Setup for Ben
#
#REPOSITORY_BASE_PATH=$HOME/git-source/Topo_LLM

# Setup for Julius
#
REPOSITORY_BASE_PATH=$HOME/Documents/LLM-Analysis/Topo_LLM

# These lines add the environment variable to the .bashrc and .zshrc files,
# so that they contain the following line:
# export TOPO_LLM_REPOSITORY_BASE_PATH=$REPOSITORY_BASE_PATH

echo "export TOPO_LLM_REPOSITORY_BASE_PATH=$REPOSITORY_BASE_PATH" >> $HOME/.bashrc
echo "export TOPO_LLM_REPOSITORY_BASE_PATH=$REPOSITORY_BASE_PATH" >> $HOME/.zshrc

