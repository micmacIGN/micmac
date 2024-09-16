#!/usr/bin/env bash

######### SET-UP
mm3dBin=$(which MMVII)
DenseMDir=${mm3dBin::(-9)}"src/DenseMatch/"
CodeDir=${DenseMDir}"RAFT-Stereo/"

# path to your trained model
MODELPATH=${CodeDir}"models/iraftstereo_rvc.pth"

# disparity scale of the trained model
#DISPSCALE=-256.0
########


# enter virtual env
#source ${DenseMDir}python_env/${EnvName}/bin/activate

# run image matching
python3 ${CodeDir}demo.py --restore_ckpt ${MODELPATH} --context_norm instance "$@"

# de-normalise disparities
#mv ${6} ${6}_unnorm.tif
#mm3d Nikrup "/ ${6}_unnorm.tif ${DISPSCALE}" ${6} "@ExitOnBrkp"


# quit virtual env
#deactivate
