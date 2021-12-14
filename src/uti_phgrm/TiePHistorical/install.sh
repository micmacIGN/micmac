#!/usr/bin/env bash

#Introduction: https://www.youtube.com/watch?v=N5vscPTWKOk

PYTHON_PATH=$1 
echo $PYTHON_PATH
PYTHON_ENV=python_env
MAGICLEAP_ENV=magicleap_env

#dependances
pip3 install virtualenv

#Clone SuperGluePretrainedNetwork
git clone https://github.com/magicleap/SuperGluePretrainedNetwork
cd SuperGluePretrainedNetwork

#create new branch 'test', resetted to last commit of 2020
git checkout -b test
git reset --hard c0626d58c843ee0464b0fa1dd4de4059bfae0ab4

#create dir to store virtual env
cd ..
mkdir ${PYTHON_ENV}
cd ${PYTHON_ENV}

#create a new virtualenv named magicleap_env
virtualenv -p ${PYTHON_PATH} ${MAGICLEAP_ENV}

#enter new virtual env
echo source ${MAGICLEAP_ENV}/bin/activate
source ${MAGICLEAP_ENV}/bin/activate
#now the prompt shows: (python_env)
cd ..

#add modules to the env
#pip3 install numpy opencv-python torch matplotlib
pip3 install -r requirements.txt
pip3 list

#quit virtual env
deactivate
