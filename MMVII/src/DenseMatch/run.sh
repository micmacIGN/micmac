#!/usr/bin/env bash

######### SET-UP
EnvName=psmnet_env
mm3dBin=$(which MMVII)
DenseMDir=${mm3dBin::(-9)}"src/DenseMatch/"
CodeDir=${DenseMDir}"PSMNet/"

# path to your trained model
MODELPATH=${CodeDir}"models/finetune_PSMnet.tar"

# disparity scale of the trained model
DISPSCALE=-256.0
########


# enter virtual env
source ${DenseMDir}python_env/${EnvName}/bin/activate

# run image matching
python3 ${CodeDir}Test_img.py --loadmodel ${MODELPATH} "$@"

# de-normalise disparities
mv ${6} ${6}_unnorm.tif
mm3d Nikrup "/ ${6}_unnorm.tif ${DISPSCALE}" ${6} "@ExitOnBrkp"


# quit virtual env
deactivate
