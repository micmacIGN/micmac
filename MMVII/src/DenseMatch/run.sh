#!/usr/bin/env bash

# change the path to your trained model
TrainedModelFile="/home/er/Documents/d_development/stereo_matching/Vaih/models/finetune_PSMnet.tar"

EnvName=psmnet_env

mm3dBin=$(which MMVII)
DenseMDir=${mm3dBin::(-9)}"src/DenseMatch/"

CodeDir=${DenseMDir}"PSMNet/"

# enter virtual env
source ${DenseMDir}python_env/${EnvName}/bin/activate

# run programm
python ${CodeDir}Test_img.py --loadmodel ${TrainedModelFile} "$@"

# quit virtual env
deactivate

