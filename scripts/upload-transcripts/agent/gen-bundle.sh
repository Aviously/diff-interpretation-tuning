#!/bin/bash
# This should be run from the root of the repository with the virtual environment activated, like:
#    . .venv/bin/activate
#    ./scripts/upload-transcripts/agent/gen-bundle.sh

. .venv/bin/activate

rm -rf ./scripts/upload-transcripts/agent/raw-logs
rm -rf ./scripts/upload-transcripts/agent/inspect-bundle
rm -f ./scripts/upload-transcripts/agent/inspect-bundle.zip

AGENT_LOGS=(
    "weight-diff-20250512-1.7b-5000-conf-2025-s42/agent/v1.3.0"
    "weight-diff-20250512-1.7b-5000-conf-2025-s42/agent/v1.3.0-with-trigger"
    "weight-diff-20250512-4b-5000-conf-2025-s42/agent/v1.3.0-q100-minified"
    "weight-diff-20250512-4b-5000-conf-2025-s42/agent/v1.3.0-with-trigger"
    "weight-diff-20250512-8b-5000-conf-2025-s42/agent/v1.3.0"
    "weight-diff-20250512-8b-5000-conf-2025-s42/agent/v1.3.0-with-trigger"
    "weight-diff-20250514-gemma-1b-conf-2025-s42/agent/v1.3.0"
    "weight-diff-20250514-gemma-1b-conf-2025-s42/agent/v1.3.0-with-trigger"
    "weight-diff-20250514-gemma-4b-conf-2025-s42/agent/v1.3.0"
    "weight-diff-20250514-gemma-4b-conf-2025-s42/agent/v1.3.0-with-trigger"
)

for log in "${AGENT_LOGS[@]}"; do
    mkdir -p "./scripts/upload-transcripts/agent/raw-logs/${log}"
    cp -r "./data/ask-qs-to-loras/${log}/"*.eval "./scripts/upload-transcripts/agent/raw-logs/${log}/"
done

# Strip metadata from the eval logs.
python ./scripts/upload-transcripts/agent/strip_metadata.py --log-dir ./scripts/upload-transcripts/agent/raw-logs

inspect view bundle --log-dir ./scripts/upload-transcripts/agent/raw-logs --output-dir ./scripts/upload-transcripts/agent/inspect-bundle

cd ./scripts/upload-transcripts/agent/inspect-bundle && zip -r ../inspect-bundle.zip .
