#!/bin/bash

DOCKER_SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
source ${DOCKER_SCRIPT_DIR}/common.sh
docker build -t image_process --build-arg USER_ID=${D_UID} --build-arg GROUP_ID=${D_GID} --build-arg USER_NAME=${D_UNAME} ${DOCKER_SCRIPT_DIR} 
