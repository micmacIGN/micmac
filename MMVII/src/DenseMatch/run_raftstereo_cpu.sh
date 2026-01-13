#!/usr/bin/env bash

######### SET-UP
mm3dBin=$(which MMVII)
DenseMDir=${mm3dBin::(-9)}"src/DenseMatch/"
CodeDir=${DenseMDir}"RAFT-Stereo/"

# path to your trained model
MODELPATH=${CodeDir}"models/270000_raftstereo_experiment-PATCH-640.pth"
#375002_epoch_raftstereo_experiment.pth.gz"
#MODELPATH=${CodeDir}"models/raftstereo-sceneflow.pth"
# disparity scale of the trained model
#DISPSCALE=-256.0
########


# enter virtual env
#source ${DenseMDir}python_env/${EnvName}/bin/activate

# run image matching
python3 ${CodeDir}demo_cpu.py --restore_ckpt ${MODELPATH} --context_norm instance "$@"

# de-normalise disparities
#mv ${6} ${6}_unnorm.tif
#mm3d Nikrup "/ ${6}_unnorm.tif ${DISPSCALE}" ${6} "@ExitOnBrkp"


# quit virtual env
#deactivate
