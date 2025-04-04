#!/bin/bash

# Check if a commit message argument is provided
if [ -z "$1" ]; then
  echo "Error: No commit message provided."
  echo "Usage: ./git_commit_and_push.sh \"Your commit message\""
  exit 1
fi

# Store the commit message
commit_message="$1"

# Get the current branch name
current_branch=$(git rev-parse --abbrev-ref HEAD)

# Run the Git commands
git add .
git commit -m "$commit_message"

# Check if commit was successful
if [ $? -eq 0 ]; then
  echo "Commit successful. Pushing changes to $current_branch..."
  git push -u origin "$current_branch"
  
  if [ $? -eq 0 ]; then
    echo "Push successful!"
  else
    echo "An error occurred while pushing changes."
  fi
else
  echo "An error occurred during commit. Changes not pushed."
fi