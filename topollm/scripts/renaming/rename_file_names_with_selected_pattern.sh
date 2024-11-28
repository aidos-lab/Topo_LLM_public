#!/bin/bash

# Initialize dry_run flag to false
dry_run=false

# Parse command line options
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dry_run)
      dry_run=true
      shift # Remove the --dry_run argument
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Find all files in the current directory matching the pattern
find . -type f -name '*ftm=standard_lora-None*' | while read -r file; do
    # Extract the directory and generate the new filename
    dir=$(dirname "$file")
    base=$(basename "$file")
    new_base="${base//ftm=standard_lora-None/standard-None}"
    
    # Check if dry_run is enabled
    if [ "$dry_run" = true ]; then
        printf ">>> [DRY RUN] Would rename:\n$file\n->\n$dir/$new_base\n"
    else
        # Perform the renaming
        mv "$file" "$dir/$new_base"
    fi
done