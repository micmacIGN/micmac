#!/usr/bin/env bash

######### SET-UP
# path to your trained model
TrainedModelFile="/home/er/Documents/d_development/stereo_matching/Vaih/models/finetune_PSMnet.tar"

# disparity scale of the trained model
DispScale=-256.0
########

EnvName=psmnet_env
mm3dBin=$(which MMVII)
DenseMDir=${mm3dBin::(-9)}"src/DenseMatch/"
CodeDir=${DenseMDir}"PSMNet/"

# enter virtual env
source ${DenseMDir}python_env/${EnvName}/bin/activate

# run image matching
python ${CodeDir}Test_img.py --loadmodel ${TrainedModelFile} "$@"

# de-normalise disparities
mv ${6} ${6}_unnorm.tif
mm3d Nikrup "/ ${6}_unnorm.tif ${DispScale}" ${6}


# quit virtual env
deactivate
