#!/bin/bash

if [ $SUDO_USER ]; then
    D_UNAME=$SUDO_USER
    D_UID=$SUDO_UID
    D_GID=$SUDO_GID
else
    D_UNAME=`whoami`
    D_UID=`id -u`
    D_GID=`id -g`
fi