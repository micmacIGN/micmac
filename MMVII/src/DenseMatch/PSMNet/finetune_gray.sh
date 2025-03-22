#! /bin/bash

python finetune_gray.py --maxdisp 384 \
						--disp_scale 64 \
						--train_datapath /work/OT/ai4geo/users/tengw/stereodensematchingbenchmark/IARPA-stereo_echo/training/IARPA_trainlist_small.txt \
                        --val_datapath /work/OT/ai4geo/users/tengw/stereodensematchingbenchmark/IARPA-stereo_echo/training/IARPA_vallist_small.txt \
						--model stackhourglass \
						--epochs 1000 \
						--loadmodel /home/qt/tengw/scratch/IARPA_Model_save/PSMNet_disp/finetune_144.tar \
						--resume \
                        --epoch_start 145 \
						--savemodel /home/qt/tengw/scratch/IARPA_Model_save/PSMNet_disp/

