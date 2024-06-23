#!/bin/bash

# Set the directory to clean (replace with your actual directory path)
target_dir="wandb"

# Safety check - Exit if the directory doesn't exist
if [ ! -d "$target_dir" ]; then
  echo "Error: Directory '$target_dir' does not exist."
  exit 1
fi
targets =  "wandb/run-20240606.*"
# Loop through all entries in the directory
for entry in "$target_dir"/*; do
  # Check if it's a directory
  if [ -d "$entry" ]; then
#      echo "$entry"
# Check if the name starts with "run-20240606"
    if [[ "$entry" =~ $regex_pattern ]]; then
 #     echo "Deleting: $entry"
      rm -rf "$entry"  # Delete the directory and its contents
    fi
  fi
done

echo "Cleaning completed."
