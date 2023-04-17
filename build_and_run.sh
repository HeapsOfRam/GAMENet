#!/bin/bash

VERS=$1
IMAGE=ryangamenet

# comment this out to use docker instead of podman
alias docker=podman

docker build -t $IMAGE:$VERS .
docker run --privileged --gpus all -it --rm $IMAGE:$VERS
