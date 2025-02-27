from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
import cv2
# a bug in PIL: cannot write mode I;16 as PNG
from PIL import Image

import pdb
import imageio
import tifffile

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default='./trained/pretrained_model_KITTI2015.tar',
                    help='loading model')
parser.add_argument('--leftimg', default= './left.png',
                    help='left image')
parser.add_argument('--rightimg', default= './right.png',
                    help='riaght image')                                      
parser.add_argument('--result', default= './Test_disparity.png',
                    help='save disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--disp_scale', type=int ,default=256,
                    help='maxium disparity') 
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cuda=False #args.no_cuda
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model)
if args.cuda:
    model.cuda()
print("CUDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA   ",args.cuda)
# ERupnik modif to adapt to cpu
device ="cpu" # torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
if args.loadmodel is not None:
    print('load PSMNet')
    state_dict = torch.load(args.loadmodel,map_location=device)
    model.load_state_dict(state_dict['state_dict'])
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
    model.eval()

    if args.cuda:
       imgL = imgL.cuda()
       imgR = imgR.cuda()     

    with torch.no_grad():
        disp = model(imgL,imgR)

    disp = torch.squeeze(disp)
    pred_disp = disp.data.cpu().numpy()

    return pred_disp


def main():
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    
    #normal_mean_var = {'mean': [0.485, 0.485, 0.485],
    #                    'std': [0.229, 0.229, 0.229]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(**normal_mean_var)])    

    #imgL_o = Image.open(args.leftimg).convert('RGB')
    #imgR_o = Image.open(args.rightimg).convert('RGB')
    imgL_o=tifffile.imread(args.leftimg)
    imgR_o=tifffile.imread(args.rightimg)
    if imgL_o.ndim<3:
        imgL_o=np.expand_dims(imgL_o,-1)
        imgL_o=np.tile(imgL_o,(1,1,3)).astype(np.uint8)
    else:
        imgL_o=imgL_o[...,0:3]
        imgL_o=imgL_o.astype(np.uint8)
    if imgR_o.ndim<3:
        imgR_o=np.expand_dims(imgR_o,-1)
        imgR_o=np.tile(imgR_o,(1,1,3)).astype(np.uint8)
    else:
        imgR_o=imgR_o[...,0:3]
        imgR_o=imgR_o.astype(np.uint8)


    imgL = infer_transform(imgL_o)
    imgR = infer_transform(imgR_o) 
    print("shape of tensor after transform ",imgL.shape) 

    # pad to width and hight to 16 times
    if imgL.shape[1] % 16 != 0:
        times = imgL.shape[1]//16       
        top_pad = (times+1)*16 -imgL.shape[1]
    else:
        top_pad = 0

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16                       
        right_pad = (times+1)*16-imgL.shape[2]
    else:
        right_pad = 0    

    imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
    imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

    start_time = time.time()
    pred_disp = test(imgL,imgR)
    print('time = %.2f' %(time.time() - start_time))

    if top_pad !=0 and right_pad != 0:
        img = pred_disp[top_pad:,:-right_pad]
    else:
        img = pred_disp

    img = (img*args.disp_scale).astype('uint16')
    imageio.imwrite(args.result, img)

if __name__ == '__main__':
   main()
