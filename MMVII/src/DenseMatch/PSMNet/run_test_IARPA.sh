#!/bin/bash

modelname="/home/qt/tengw/scratch/IARPA_Model_save/PSMNet_disp/finetune_158.tar"

data_path="/home/qt/tengw/scratch/IARPA-MVS/disp_full/testing/02APR15WV031000015APR02134718-P1BS-500497282050_01_P001_________AAE_0AAAAABPABJ0_02APR15WV031000015APR02134802-P1BS-500276959010_02_P001_________AAE_0AAAAABPABC0"
SaveDir="/home/qt/tengw/scratch/IARPA-MVS/experiment_compare/"

file="02APR15WV031000015APR02134718-P1BS-500497282050_01_P001_________AAE_0AAAAABPABJ0_02APR15WV031000015APR02134802-P1BS-500276959010_02_P001_________AAE_0AAAAABPABC0_0000.png"

limg="${data_path}/colored_0/${file}"
rimg="${data_path}/colored_1/${file}"

python Test_gray.py --loadmodel ${modelname}  \
				    --leftimg ${limg} \
				    --rightimg ${rimg} \
				    --result ${resultDir}${file} \
				    --disp_scale 64

