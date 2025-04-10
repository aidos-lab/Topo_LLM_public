#!/bin/bash

# This script sets up the environment variables for the repository.
#
# Note: This script only needs to be run once. It sets up the environment
# variables in the .bashrc and .zshenv files in the home directory.

echo ">>> Setting up environment variables for repository ..."

# Automatically infer the repository base path two levels above the current shell script
REPOSITORY_BASE_PATH="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"

# Uncomment and modify the following line to manually set the repository base path.
#
# REPOSITORY_BASE_PATH=$HOME/git-source/grokking

REPOSITORY_BASE_PATH_VARIABLE_NAME="TOPO_LLM_REPOSITORY_BASE_PATH"

echo ">>> The repository base path is:"
echo ">>> $REPOSITORY_BASE_PATH"
echo ">>> The environment variable name is:"
echo ">>> $REPOSITORY_BASE_PATH_VARIABLE_NAME"

# Check whether the repository base path is already set
if grep -q "$REPOSITORY_BASE_PATH_VARIABLE_NAME" $HOME/.bashrc; then
    echo "@@@ The environment variable is already set in $HOME/.bashrc"

    echo "@@@ The current line in $HOME/.bashrc is:"
    grep "$REPOSITORY_BASE_PATH_VARIABLE_NAME" $HOME/.bashrc
    echo "@@@ The current line in $HOME/.zshenv is:"
    grep "$REPOSITORY_BASE_PATH_VARIABLE_NAME" $HOME/.zshenv

    echo "@@@ This script will not overwrite the current value."
    echo "@@@ If you want to change the value, please do so manually."
    echo "@@@ The script will exit now."

    exit 1
else
    echo ">>> The environment variable is not set in $HOME/.bashrc"
fi

# These lines add the environment variable to the .bashrc and .zshenv files, so that they contain the following line:
# export $REPOSITORY_BASE_PATH_VARIABLE_NAME=$REPOSITORY_BASE_PATH

LINE_TO_ADD="export $REPOSITORY_BASE_PATH_VARIABLE_NAME=\"$REPOSITORY_BASE_PATH\""
echo ">>> The following line will be added to $HOME/.bashrc and $HOME/.zshenv:"
echo ">>> $LINE_TO_ADD"

echo ">>> Writing to $HOME/.bashrc ..."
echo $LINE_TO_ADD >>$HOME/.bashrc

echo ">>> Writing to $HOME/.zshenv ..."
echo $LINE_TO_ADD >>$HOME/.zshenv

echo ">>> Setting up environment variables for repository DONE"
