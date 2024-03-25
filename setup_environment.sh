# export TOPO_LLM_REPOSITORY_BASE_PATH=$HOME/git-source/Topo_LLM

echo "Setting up environment variables for Topo_LLM"

# Setup for Ben
#
#REPOSITORY_BASE_PATH=$HOME/git-source/Topo_LLM

# Setup for Julius
#
REPOSITORY_BASE_PATH=$HOME/Documents/LLM-Analysis/Topo_LLM

echo "export TOPO_LLM_REPOSITORY_BASE_PATH=$REPOSITORY_BASE_PATH" >> $HOME/.bashrc
echo "export TOPO_LLM_REPOSITORY_BASE_PATH=$REPOSITORY_BASE_PATH" >> $HOME/.zshrc

