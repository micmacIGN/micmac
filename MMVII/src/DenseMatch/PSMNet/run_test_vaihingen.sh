#!/bin/bash

modelname="/home/qt/tengw/scratch/Vaihingen_Model_save/PSMNet_scratch/finetune_363.tar"
data_path="/work/OT/ai4geo/users/tengw/stereodensematchingbenchmark/Vaihingen-stereo/testing/10030062_10030063"
SaveDir="/home/qt/tengw/scratch/vaihingen/experiment_scratch/"


file="10030062_10030063_0000.png"
limg="${data_path}/colored_0/${file}"
rimg="${data_path}/colored_1/${file}"

python Test_img.py --loadmodel ${modelname}  \
				   --leftimg ${limg} \
				   --rightimg ${rimg} \
				   --result ${resultDir}${file}

