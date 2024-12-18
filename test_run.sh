#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_TAG="example-algorithm"
DOCKER_NOOP_VOLUME="${DOCKER_TAG}-volume"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"
MODEL_DIR="${SCRIPT_DIR}/test/opt/ml/model"


cleanup() {
    echo "=+=  Cleaning permissions ..."
    # Ensure permissions are set correctly on the output
    # This allows the host user (e.g. you) to access and handle these files
    docker run --rm \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      $DOCKER_TAG \
      -c "chmod -R -f o+rwX /output/* || true"
}

echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/build.sh" "$DOCKER_TAG"


if [ -d "$OUTPUT_DIR" ]; then
  # Ensure permissions are setup correctly
  # This allows for the Docker user to write to this location

  rm -rf "${OUTPUT_DIR}"/*
  chmod -f o+rwx "$OUTPUT_DIR"
else
  mkdir -m o+rwx "$OUTPUT_DIR"
fi

trap cleanup EXIT

echo "=+= Doing a forward pass"
## Note the extra arguments that are passed here:
# '--network none'
#    entails there is no internet connection
# '--volume <NAME>:/tmp'
#   is added because on Grand Challenge this directory cannot be used to store permanent files
docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null

docker run --rm \
    --platform=linux/amd64 \
    --network none \
    --volume "$INPUT_DIR":/input:ro \
    --volume "$OUTPUT_DIR":/output \
    --volume "$MODEL_DIR":/opt/ml/model \
    --volume "$DOCKER_NOOP_VOLUME":/tmp \
    $DOCKER_TAG

docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null

echo "=+= Wrote results to ${OUTPUT_DIR}"

echo "=+= Save this image for uploading via ./save.sh"
