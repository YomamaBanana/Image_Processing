#!/bin/bash

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
cd $SCRIPT_DIR
VENV=${SCRIPT_DIR}/.venv
mkdir ${VENV}

python3 -m virtualenv ${VENV} --python=python3
. ${VENV}/bin/activate
python3 -m pip install --upgrade pip

python3 -m pip install -r requirements.txt
