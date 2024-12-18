#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_TAG="example-algorithm"


# Check if an argument is provided
if [ "$#" -eq 1 ]; then
    DOCKER_TAG="$1"
fi

# Note: the build-arg is JUST for the workshop
docker build "$SCRIPT_DIR" \
  --platform=linux/amd64 \
  --build-arg BUILD_VERSION=$(date +%s) \
  --tag $DOCKER_TAG 2>&1