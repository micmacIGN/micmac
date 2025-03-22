#! /bin/bash


python finetune_vaihingen.py --maxdisp 192 \
							 --train_datapath /work/OT/ai4geo/users/tengw/stereodensematchingbenchmark/Vaihingen-stereo/training/vaihingen_trainlist.txt \
                             --val_datapath /work/OT/ai4geo/users/tengw/stereodensematchingbenchmark/Vaihingen-stereo/training/vaihingen_vallist.txt \
							 --model stackhourglass \
							 --loadmodel /home/qt/tengw/scratch/Aerial_Model_Save/PSMNet_large/finetune_208.tar \
							 --resume \
                             --epoch_start 209 \
							 --epochs 1000 \
							 --savemodel /home/qt/tengw/scratch/Aerial_Model_Save/PSMNet_large/

