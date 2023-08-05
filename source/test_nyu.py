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

# #f in mm
# train_f=float(args.eval_trained_rgb_dir.split('_')[-3])
# #fdist in m
# train_fdist=float(args.eval_trained_rgb_dir.split('_')[-1])

# test_f=float(args.eval_test_rgb_dir[0].split('_')[-3])
# test_fdist=float(args.eval_test_rgb_dir[0].split('_')[-1])

# kcam=train_f**2/test_f**2 * (test_fdist-test_f*1e-3)


result=test.validate_dist(val_loader,model,criterion,0,args,min_dist=0.0,max_dist=2.0)
print(result)

# for n in range(0,100):
#     kcam=n/10.
#     print('kcam='+str(kcam))
#     result=test.validate_dist(val_loader,model,criterion,0,args,min_dist=0.0,max_dist=2.0,kcam=kcam)
#     print(result)
