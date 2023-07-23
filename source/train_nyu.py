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
train_dataset=nyudepthv2.nyudepthv2(data_path=args.data_path,rgb_dir=args.rgb_dir,depth_dir=args.depth_dir,crop_size=crop_size,is_blur=args.is_blur,is_train=True)
val_dataset=nyudepthv2.nyudepthv2(data_path=args.data_path,rgb_dir=args.rgb_dir,depth_dir=args.depth_dir,crop_size=crop_size,is_blur=args.is_blur,is_train=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                           num_workers=0,pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                         num_workers=0,pin_memory=True)

'''
load defocusNet model
'''
device_id=0
ch_inp_num = 3
ch_out_num = 1
model = AENet(ch_inp_num, 1, 16, flag_step2=True).to(device_id)
if args.resume_from:
    pretrained_dict = torch.load(args.resume_from)
    model.load_state_dict(pretrained_dict['state_dict'])


model_params = model.parameters()
criterion=torch.nn.MSELoss()
optimizer = optim.Adam(model_params,lr=args.min_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)
model.train()

evalitr=10
for i in range(600):
    total_d_loss,total_b_loss=0,0
    for batch_idx, batch in enumerate(train_loader):
        input_RGB=batch['image'].float().to(device_id)
        depth_gt=batch['depth'].to(device_id)
        class_id=batch['class_id']
        gt_blur=batch['blur'].to(device_id)
        fdist=batch['fdist'].to(device_id)
        f=batch['f'].to(device_id)
        scale=f**2
        kcam=(1/(f**2)*scale).float()

        optimizer.zero_grad()

        mask=(depth_gt>0.0)*(depth_gt<2.0).detach_()

        depth_pred,blur_pred = model(input_RGB,flag_step2=True,kcam=0)

        loss_d=criterion(depth_pred.squeeze(dim=1)[mask], depth_gt[mask])
        loss_b=criterion(blur_pred.squeeze(dim=1)[mask],gt_blur[mask])
        if(torch.isnan(loss_d) or torch.isnan(loss_b)):
            # print('nan in losses')
            # logging.info('nan in losses')
            continue

        loss=loss_d+loss_b
        total_d_loss+=loss_d.item()
        total_b_loss+=loss_b.item()
        loss.backward()
        optimizer.step()
    
    optimizer.step()
    print("Epochs=%3d blur loss=%5.4f  depth loss=%5.4f" %(i,total_b_loss/len(train_loader),total_d_loss/len(train_loader)))  
    logging.info("Epochs=%3d blur loss=%5.4f  depth loss=%5.4f" , i,total_b_loss/len(train_loader),total_d_loss/len(train_loader))

    if (i+1)%evalitr==0:
        with torch.no_grad():
            result=test.validate(val_loader, model, criterion, device_id, args)
            print(result)
            model.train()
