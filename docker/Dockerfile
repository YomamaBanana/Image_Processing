FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=ubuntu

ENV USER ${USER_NAME}
ENV HOME /home/${USER}

RUN groupadd  -g $GROUP_ID $USER
RUN useradd -m -g $GROUP_ID -G sudo -u $USER_ID $USER

WORKDIR ${HOME}

RUN apt-get update
RUN apt-get -y install wget git
RUN apt-get -y install build-essential cmake
RUN apt-get -y install python3 python3-dev python3-distutils python3-tk
RUN apt-get -y install openmpi-bin libopenmpi-dev
RUN apt-get -y install libopencv-dev
RUN apt-get -y install sudo

RUN echo 'Defaults visiblepw'             >> /etc/sudoers
RUN echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USER_ID}
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py
RUN python3 -m pip install virtualenv
RUN mkdir -p ${HOME}/.config/matplotlib
RUN echo "backend : TkAgg" >> ${HOME}/.config/matplotlib/matplotlibrc

ENV PATH=$PATH:${HOME}/.local/bin
