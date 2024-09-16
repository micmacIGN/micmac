# PSM net Code for HAL

## Introduction

An end to end method, multi scale feature.
<mark>Clone from [origin code](https://github.com/JiaRenChang/PSMNet).</mark>


## Code on HAL for Aerial or satellite dataset

Because in our dataset, there is aerial dataset and satellite dataset. Aerial dataset usually has three bands, and satellite is gray image.

### Traning example(Color/Gray)
**Considering the training time on HAL, each job can only run 12h, so the epoch is not that important.**
So there is two files to handle this problem, for the RGB image(3 bands), an example is ***finetune_color_vaihingen.sh***, the disparity scale is used a default value(256), training command line:

#### fine tune form  [KITTI model](https://drive.google.com/file/d/15NhbtZfMBHGsDp4NaUMzdXDqyvcOvhsu/view?usp=sharing) :

```console
#! /bin/bash

#                            # max disparity
python finetune_vaihingen.py --maxdisp 192 \
#                            # train file list
                             --train_datapath '/training/train_filelist.txt' \
#                            # test file list
                             --val_datapath '/training/val_filelist.txt' \
#                            # model name
                             --model stackhourglass \
#                            # input model
                             --loadmodel '/PSMNet_large/pretrained_model_KITTI2015.tar' \
#                            # start epoch
                             --epoch_start 0 \ 
#                            # iteration times                    
                             --epochs 1000 \
#                            # save model directory                             
                             --savemodel '/PSMNet_large/'
```

#### continue to training after a break point :

```console
#! /bin/bash

#                            # max disparity
python finetune_vaihingen.py --maxdisp 192 \
#                            # train file list
                             --train_datapath '/training/train_filelist.txt' \
#                            # test file list
                             --val_datapath '/training/val_filelist.txt' \
#                            # model name
                             --model stackhourglass \
#                            # input model
                             --loadmodel '/PSMNet_large/finetune_208.tar' \
#                            # load the optimizer paraters at the same time                 
                             --resume \
#                            # start epoch
                             --epoch_start 209 \ 
#                            # iteration times                    
                             --epochs 1000 \
#                            # save model directory                             
                             --savemodel '/PSMNet_large/'
```
For the gray image, an example is ***finetune_gray.sh***, the comand line of continue to training after a break point:

```console
#! /bin/bash

#                       # max disparity
python finetune_gray.py --maxdisp 384 \
#                       # ground truth disparity scale
                        --disp_scale 64 \
#                       # train file list
                        --train_datapath '/training/train_filelist.txt' \
#                       # test file list                        
                        --val_datapath '/training/val_filelist.txt' \
#                       # model name                        
                        --model stackhourglass \
#                       # iteration times                        
                        --epochs 1000 \
#                       # input model                        
                        --loadmodel '/PSMNet_disp/finetune_144.tar' \
#                       # load the optimizer paraters at the same time                       
                        --resume \
#                       # start epoch
                        --epoch_start 145 \ 
#                       # save model directory                         
                        --savemodel '/PSMNet_disp/'
```

### Testing example(Color/Gray)
For the test, an example for RGB image is ***run_test_vaihingen.sh***, the command line:
```console
#! /bin/bash
#                  # input model 
python Test_img.py --loadmodel ${modelname}  \
#                  # left image
                   --leftimg ${limg} \
#                  # right image
                   --rightimg ${rimg} \
#                  # result
                   --result ${SaveDir}${file}
```

An example for gray image is ***run_test_IARPA.sh***, the command line:

```console
#! /bin/bash
#                   # input model 
python Test_gray.py --loadmodel ${modelname}  \
#                   # left image
                    --leftimg ${limg} \
#                   # right image
                    --rightimg ${rimg} \
#                   # result
                    --result ${SaveDir}${file} \
#                   # result disparity scale              
                    --disp_scale 64
```

### Evaluation full dataset

At the same time, after generate the file list for all the dataset, you can also evaluate at one time.

For the color dataset, an example is shown in ***evaluate_color.sh***,  the disparity result is saved in path **${SaveDir}/pair_name/${Method}/**, the command line:
```console
#! /bin/bash
#                       # input model 
python evluate_color.py --loadmodel ${modelname}  \
#                       # file list 
                        --datalist ${LIST} \
#                       # save disparity path
                        --savepath ${SaveDir} \
#                       # folder name
                        --subfolder ${Method}
```
For the gray, an example is shown in ***evaluate_gray.sh***,  the disparity result is saved in path **${SaveDir}/pair_name/${Method}/**, the command line:
```console
#! /bin/bash
#                      # input model 
python evluate_gray.py --loadmodel ${modelname}  \
#                      # file list 
                       --datalist ${LIST} \
#                      # save disparity path
                       --savepath ${SaveDir} \
#                      # folder name
                       --subfolder ${Method} \
#                      # result disparity scale  
                       --disp_scale 64
```

In the examples, the file list can be found in test data folder, refer to ***vaihingen_test.txt*** in [data introduction](https://github.com/whuwuteng/benchmark_ISPRS2021).

## Training Model

### Pre-tained on KITTI

This is only for the fine-tune using this code, you can fine-tune on [KITTI model](https://drive.google.com/file/d/15NhbtZfMBHGsDp4NaUMzdXDqyvcOvhsu/view?usp=sharing).

### Pre-trained on Vaihingen

At present, some model can be trained from Vaihigen dataset, the iteration number is **500**, you can download from [GoogleDrive](https://drive.google.com/file/d/1TMG-gGwb0e417QbiLCGLB9DV0k0hHA5b/view?usp=sharing).

## Feed Back

If you think you have any problem, contact [Teng Wu] <Teng.Wu@ign.fr>
