#!/bin/bash

# Check if TOPO_LLM_REPOSITORY_BASE_PATH is set
if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo "Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  exit 1
fi

echo "📂 TOPO_LLM_REPOSITORY_BASE_PATH=${TOPO_LLM_REPOSITORY_BASE_PATH}"

echo "⏳ Loading environment variables from .env file..."
source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# # # # # # # # # # # #
# Directory to backup
SOURCE_DIR="${TOPO_LLM_REPOSITORY_BASE_PATH}/hydra_output_dir"

# Get absolute path of source directory
ABSOLUTE_SOURCE_DIR=$(realpath "$SOURCE_DIR")

# Get current timestamp
TIMESTAMP=$(date +%Y-%m-%d.%H%M%S)

# Create backup filename
BACKUP_FILE="${SOURCE_DIR}.backup.${TIMESTAMP}.tar.gz"
# Calculate absolute path manually without requiring file to exist
ABSOLUTE_BACKUP_FILE="$(dirname "$ABSOLUTE_SOURCE_DIR")/$(basename "$BACKUP_FILE")"

# Display information before starting
echo "📂 Source directory: $ABSOLUTE_SOURCE_DIR"
echo "📦 Target backup file: $BACKUP_FILE"
echo "📦 Absolute target backup file: $ABSOLUTE_BACKUP_FILE"
echo "⏳ Starting backup process..."

# Create compressed archive with verbose output to show progress
echo "⏳ Creating backup archive..."
tar -czvf "$BACKUP_FILE" "$SOURCE_DIR"

# Get the size of the resulting archive
BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)

echo "✅ Backup created: $BACKUP_FILE"
echo "📊 Backup size: $BACKUP_SIZE"

echo "✅ Exiting script."
exit 0