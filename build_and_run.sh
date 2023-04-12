#!/bin/bash

VERS=$1
IMAGE=ryangamenet

podman build -t $IMAGE:$VERS .
podman run --privileged --gpus all -it --rm $IMAGE:$VERS
