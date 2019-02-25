#!/bin/bash

CONTAINER_NAME=$1
if [[ -z "${CONTAINER_NAME}" ]]; then
    echo "Missing CONTAINER_NAME"
    echo "Example usage: ./run_image.sh tensorflow:latest-gpu /dir /"
    exit(1)
fi

HOST_DATA_DIR=$2
if [[ -z "${HOST_DATA_DIR}" ]]; then
  echo "Missing HOST_DATA_DIR"
  echo "Example usage: ./run_image.sh tensorflow:latest-gpu /dir"
  exit(1)
fi

CONTAINER_DATA_DIR=$3
if [[ -z "${CONTAINER_DATA_DIR}" ]]; then
    CONTAINER_DATA_DIR=~/
fi

echo "Container name    : ${CONTAINER_NAME}"
echo "Host data dir     : ${HOST_DATA_DIR}"
echo "Container data dir: ${CONTAINER_DATA_DIR}"
CONTAINER_ID=`docker ps -aqf "name=^/${CONTAINER_NAME}$"`
if [ -z "${CONTAINER_ID}" ]; then
  echo "Creating new docker container."
  nvidia-docker run -it --privileged --network=host -v ${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}:rw --name=${CONTAINER_NAME} streetview:gpu bash
else
  echo "Found running container: ${CONTAINER_ID}."
  # Check if the container is already running and start if necessary.
  if [ -z `docker ps -qf "name=^/${CONTAINER_NAME}$"` ]; then
      echo "Starting and attaching to ${CONTAINER_NAME} container..."
      docker start ${CONTAINER_ID}
      docker attach ${CONTAINER_ID}
  else
      echo "Found running ${CONTAINER_NAME} container, attaching bash..."
      docker exec -it ${CONTAINER_ID} bash
  fi

fi
