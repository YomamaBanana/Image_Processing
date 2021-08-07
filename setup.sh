#!/bin/bash

# build docker
pushd docker
sudo -E bash build.sh
popd
sudo docker run -ti --rm \
    --volume="$(cd $(dirname ${BASH_SOURCE:-$0}); pwd):/home/${SUDO_USER:-$USER}/ImageProcessing" \
    --net=host \
    image_process /bin/bash ~/ImageProcessing/env/setup_env.sh


