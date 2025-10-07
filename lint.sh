#!/bin/bash

echo "Running ruff lint checks..."
ruff check
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  echo "To try and automatically fix lint issues, run 'ruff check --fix'"
fi

echo
echo "Running ruff format checks..."
ruff format --check
FORMAT_EXIT_CODE=$?
if [ $FORMAT_EXIT_CODE -ne 0 ]; then
  echo "To try and automatically fix format issues, run 'ruff format'"
fi

if [ $LINT_EXIT_CODE -ne 0 ] || [ $FORMAT_EXIT_CODE -ne 0 ]; then
  echo
  echo "Lint or format checks failed. Please fix the issues and try again."
  exit 1
fi
