#!/bin/bash

DOCKER_SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
source ${DOCKER_SCRIPT_DIR}/common.sh

echo $D_UNAME

xhost + > /dev/null 2>&1
xhost local: > /dev/null 2>&1
# docker run --gpus='"device=1"' -ti --rm \
docker run -ti --rm \
    --volume="${DOCKER_SCRIPT_DIR}/..:/home/${D_UNAME}/ImageProcessing" \
    --net=host \
    --env=DISPLAY=$DISPLAY \
    --env=QT_X11_NO_MITSHM=1 \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$HOME/.Xauthority:/home/`whoami`/.Xauthority:rw" \
    image_process $1

