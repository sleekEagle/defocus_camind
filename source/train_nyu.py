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
train_dataset=nyudepthv2.nyudepthv2(data_path=args.data_path,rgb_dir_list=args.rgb_dir,depth_dir=args.depth_dir,crop_size=crop_size,is_blur=args.is_blur,is_train=True)
val_dataset=nyudepthv2.nyudepthv2(data_path=args.data_path,rgb_dir_list=args.rgb_dir,depth_dir=args.depth_dir,crop_size=crop_size,is_blur=args.is_blur,is_train=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                           num_workers=0,pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                         num_workers=0,pin_memory=True)

logger.info(args.rgb_dir)
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
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=350, gamma=0.1)
model.train()

evalitr=10
'''
virtual_bs is used when the dataloader bs is 1 but 
we only step and zero the optimizer every virtual_bs steps so it acts as if the bs is larger
This is helpfull when the GPU is not sufficient to fit a larger batch 
but we can use a larger effective batch size
'''
virtual_bs=12
base_f=25*1e-3
for i in range(800):
    total_d_loss,total_b_loss=0,0
    for batch_idx, batch in enumerate(train_loader):
        input_RGB=batch['image'].float().to(device_id)
        depth_gt=batch['depth'].to(device_id)
        class_id=batch['class_id']
        gt_blur=batch['blur'].to(device_id)
        f=batch['f']
        fdist=batch['fdist']
        kcam=(fdist-f)*(base_f**2)/(f**2)
        kcam=torch.unsqueeze(kcam,dim=1).unsqueeze(dim=1)
        kcam=torch.repeat_interleave(kcam,dim=1,repeats=input_RGB.shape[-2])
        kcam=torch.repeat_interleave(kcam,dim=2,repeats=input_RGB.shape[-1])
        kcam=(kcam.to(device_id)).float()
        kcam=torch.unsqueeze(kcam,dim=1)   

        mask=(depth_gt>0.0)*(depth_gt<2.0).detach_()

        if args.is_kcam==0:
            kcam=0
        depth_pred,blur_pred = model(input_RGB,flag_step2=True,kcam=kcam)

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
    
        if batch_idx%virtual_bs==0:
            # print('optimizeinv : batch idx:'+str(batch_idx))
            optimizer.step()
            optimizer.zero_grad()

    print("Epochs=%3d blur loss=%5.4f  depth loss=%5.4f" %(i,total_b_loss/len(train_loader),total_d_loss/len(train_loader)))  
    logging.info("Epochs=%3d blur loss=%5.4f  depth loss=%5.4f" , i,total_b_loss/len(train_loader),total_d_loss/len(train_loader))

    if (i+1)%evalitr==0:
        with torch.no_grad():
            result=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=0.0,max_dist=2.0,kcam=kcam)
            print(result)
            logging.info(result)
            modelname="_".join([str(element)[10:] for element in args.rgb_dir])
            torch.save({
                'state_dict': model.state_dict()
                },  os.path.join(os.path.abspath(args.resultspth),(modelname)+'.tar'))
            
            model.train()
