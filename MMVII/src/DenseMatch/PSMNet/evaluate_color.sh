#!/bin/bash

modelname="/home/qt/tengw/scratch/Vaihingen_Model_echo/PSMNet_color/finetune_165.tar"
Method="PSMnet"
data_path="/work/OT/ai4geo/users/tengw/stereodensematchingbenchmark/Vaihingen-stereo/testing/"
SaveDir="/home/qt/tengw/scratch/vaihingen/experiment_echo"
LIST="${data_path}vaihingen_test.txt"

python evalue_color.py --loadmodel ${modelname}  \
--datalist ${LIST} \
--savepath ${SaveDir} \
--subfolder ${Method}
