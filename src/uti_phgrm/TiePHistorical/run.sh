#!/usr/bin/env bash

# change SGDir if your SuperGluePretrained network is not stored in MicMac dir
#mm3dBin="/etc/opt/histopipe/bin/mm3d"

mm3dBin=$(which mm3d)
TPHistoDir=${mm3dBin::(-8)}"src/uti_phgrm/TiePHistorical/"
SGDir=${TPHistoDir}"SuperGluePretrainedNetwork/"

# enter virtual env
source ${TPHistoDir}python_env/magicleap_env/bin/activate

# run programm
${SGDir}match_pairs.py "$@"

# quit virtual env
deactivate
