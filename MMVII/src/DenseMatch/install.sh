#!/usr/bin/env bash

#Introduction: https://www.youtube.com/watch?v=N5vscPTWKOk

PYTHON_PATH=/etc/opt/anaconda3/bin/python
PYTHON_ENV=python_env
THIS_ENV=psmnet_env

#dependances
pip3 install virtualenv

#Clone PSMNet
git clone https://github.com/erupnik/PSMNet.git
cd PSMNet

#create new branch 'test', resetted to initial commit
git checkout -b test
git reset --hard 9ba1e36903f3ba2c99e5be8f03d31d2751a2cb33

#create dir to store virtual env
cd ..
mkdir ${PYTHON_ENV}
cd ${PYTHON_ENV}

#create a new virtualenv named psmnet_env
virtualenv -p ${PYTHON_PATH} ${THIS_ENV}

#enter new virtual env
echo source ${THIS_ENV}/bin/activate
source ${THIS_ENV}/bin/activate
#now the prompt shows: (python_env)
cd ..

#add modules to the env
pip3 install -r requirements.txt
pip3 list

#quit virtual env
deactivate

