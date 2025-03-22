#!/bin/bash

modelname="/home/qt/tengw/scratch/IARPA_Model_echo/PSMNet_disp/finetune_165.tar"
Method="PSMnet"
data_path="/work/OT/ai4geo/users/tengw/stereodensematchingbenchmark/IARPA-stereo_echo/testing/"
SaveDir="/home/qt/tengw/scratch/IARPA-MVS/experiment_echo"
LIST="${data_path}iarpa_test.txt"

python evluate_gray.py --loadmodel ${modelname}  \
--datalist ${LIST} \
--savepath ${SaveDir} \
--subfolder ${Method} \
--disp_scale 64
