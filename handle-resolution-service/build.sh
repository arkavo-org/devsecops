#!/bin/bash
set -e

# Ensure we're using Python 3.12
PYTHON=python3.12

# Install dependencies to the Lambda layer directory
$PYTHON -m pip install -r requirements.txt -t dependencies/python

# Clean up unnecessary files
find dependencies/python -type d -name "tests" -exec rm -rf {} +
find dependencies/python -type d -name "__pycache__" -exec rm -rf {} +
find dependencies/python -type f -name "*.pyc" -delete
find dependencies/python -type f -name "*.pyo" -delete
find dependencies/python -type f -name "*.dist-info" -exec rm -rf {} +

# Build SAM application
sam build

# Deploy (optional, can be run separately)
# sam deploy --guided