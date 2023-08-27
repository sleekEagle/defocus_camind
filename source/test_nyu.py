#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:54:12 2018
@author: maximov
"""

import torch
import torch.optim as optim
import torch.utils.data

from dataloaders import nyudepthv2
from arch.AENET import AENet
from configs.nyu_options import NYUOptions
import torch.optim as optim
import logging
logger=logging
import os
from os.path import join
from datetime import datetime
from utils_depth import test

opt = NYUOptions()
args = opt.initialize().parse_args()
print(args)

#setting up logging
if not os.path.exists(args.resultspth):
    os.makedirs(args.resultspth)
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H_%M_%S_camind_NYU")+'.log'
logpath=join(args.resultspth,dt_string)
logger.basicConfig(filename=logpath,filemode='w', level=logger.INFO)
logger.info('Starting training')
logger.info(args)

#get dataloaders
crop_size=(args.crop_h, args.crop_w)
val_dataset=nyudepthv2.nyudepthv2(data_path=args.data_path,rgb_dir_list=args.eval_test_rgb_dir,depth_dir=args.depth_dir,crop_size=crop_size,is_blur=args.is_blur,is_train=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                         num_workers=0,pin_memory=True)

'''
load defocusNet model
'''
device_id=0
ch_inp_num = 3
ch_out_num = 1
model = AENet(ch_inp_num, 1, 16, flag_step2=True).to(device_id)
if args.trained_model:
    pretrained_dict = torch.load(args.trained_model)
    model.load_state_dict(pretrained_dict['state_dict'])


model_params = model.parameters()
criterion=torch.nn.MSELoss()
model.eval()


# result=test.validate_dist(val_loader,model,criterion,0,args,min_dist=0.0,max_dist=2.0,kcam_exp=0.753)
# print(result)

import numpy as np
# f_30 =np.array([2.25751742, 2.11970182, 1.99774468, 1.88905766, 1.79158664,
#        1.70368064, 1.62399757, 1.5514352 , 1.48507985, 1.42416777,
#        1.36805556, 1.31619738, 1.26812714, 1.22344442, 1.18180335,
#        1.14290356, 1.10648298, 1.07231193, 1.04018823, 1.00993323,
#        0.98138849])

f_40=np.array([0.98220013, 0.95518059, 0.92960782, 0.90536865, 0.88236142,
       0.86049452, 0.83968524, 0.81985865, 0.80094675, 0.78288767,
       0.765625  , 0.74910719, 0.73328704, 0.71812128, 0.70357012,
       0.68959694, 0.67616798, 0.66325205, 0.6508203 , 0.63884601])


for kcam in f_40:
    print('kcam='+str(kcam))
    result=test.validate_dist(val_loader,model,criterion,0,args,min_dist=0.0,max_dist=2.0,kcam_exp=kcam)
    print(result)
