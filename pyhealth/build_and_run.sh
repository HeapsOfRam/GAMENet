#!/bin/bash

VERS=$1
IMAGE=pyhealthgamenet
MIMIC=${2:-"4"}
#: ${MIMIC:=4}

echo $MIMIC

alias docker=podman

#docker build -t $IMAGE:$VERS --build-arg MIMIC=$MIMIC .
docker build -t $IMAGE:$VERS .
#docker run --privileged --gpus all -it --rm --mount type=bind,source="$(pwd)"/hiddendata,target=/app/hiddendata/ $IMAGE:$VERS
docker run --privileged --gpus all -it --rm --mount type=bind,source="$(pwd)"/hiddendata,target=/app/hiddendata/ $IMAGE:$VERS --mimic=$MIMIC -a
