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


result=test.validate_dist(val_loader,model,criterion,0,args,min_dist=0.0,max_dist=2.0)
print(result)

# import numpy as np
# # f_30 =np.array([2.25751742, 2.11970182, 1.99774468, 1.88905766, 1.79158664,
# #        1.70368064, 1.62399757, 1.5514352 , 1.48507985, 1.42416777,
# #        1.36805556, 1.31619738, 1.26812714, 1.22344442, 1.18180335,
# #        1.14290356, 1.10648298, 1.07231193, 1.04018823, 1.00993323,
# #        0.98138849])

# f_40=np.array([0.98220013, 0.95518059, 0.92960782, 0.90536865, 0.88236142,
#        0.86049452, 0.83968524, 0.81985865, 0.80094675, 0.78288767,
#        0.765625  , 0.74910719, 0.73328704, 0.71812128, 0.70357012,
#        0.68959694, 0.67616798, 0.66325205, 0.6508203 , 0.63884601])


# for kcam in f_40:
#     print('kcam='+str(kcam))
#     result=test.validate_dist(val_loader,model,criterion,0,args,min_dist=0.0,max_dist=2.0,kcam_exp=kcam)
#     print(result)

#plot some results
import cv2
from matplotlib import pyplot as plt

base_f=25e-3
for batch_idx, batch in enumerate(val_loader):
    input_RGB=batch['image'].float().to(device_id)
    depth_gt=batch['depth'].to(device_id)
    class_id=batch['class_id']
    gt_blur=batch['blur'].to(device_id)
    f=batch['f']
    fdist=batch['fdist']
    kcam=(fdist-f)*(base_f**2)/(f**2)
    x2=fdist.tolist()
    kcam=kcam.tolist()
    
    input_RGB=input_RGB[:,:,:,0:480]
    gt_blur=gt_blur[:,:,0:480]
    depth_gt=depth_gt[:,:,0:480]

    if(batch_idx==0):
        break


with torch.no_grad():
    pred_d,pred_b =model(input_RGB,flag_step2=True,x2_list=x2,kcam_list=[1.0])


plt.imshow(input_RGB.detach().cpu().numpy()[0,0,:,:])
plt.show()
depth_gt[depth_gt>2]=0
plt.imshow(depth_gt.detach().cpu().numpy()[0,:,:])
plt.show()

plt.imshow(pred_d.detach().cpu().numpy()[0,0,:,:])
plt.show()

#test on a single image
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import math
import matplotlib.image

img_path=r'C:\Users\lahir\data\pixelcalib\telephoto\OpenCamera\crop21.jpg'
# img_path=r'C:\Users\lahir\data\kinectimgs\kinect\f_40\3_480.png'
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
transform = transforms.ToTensor()
image_t = transform(image).cuda()
image_t=torch.unsqueeze(image_t,0)
kcam=[0.3]
x2=[2.0]

for kcam in np.linspace(0.765-0.7,0.765+0.7,num=100):
    print(kcam)
    with torch.no_grad():
        fname='C:\\Users\\lahir\\data\\kinectimgs\\kinect\\f_40\\kcam_variation\\pred_kcam_'+str(round(kcam,1))+'.jpeg'
        pred_d,pred_b =model(image_t,flag_step2=True,x2_list=x2,kcam_list=kcam)
        im = pred_d.detach().cpu().numpy()[0,0,:,:]
        b = pred_b.detach().cpu().numpy()[0,0,:,:]
        # im=(im-np.min(im))/(np.max(im)-np.min(im))
        plt.imshow(im)
        plt.show()
        plt.imsave(fname, im, cmap='gray')

    plt.imshow()
    plt.show()

plt.imshow(image_t.detach().cpu().numpy()[0,0,:,:])
plt.show()
        


w,h = image_t.shape[2],image_t.shape[3]
image_num=torch.zeros((1,1,w,h))
pred_img=torch.zeros((1,1,w,h)).cuda()


#window slide
step=20
for i in range(0,w-480,step):
    for j in range(0,h-480,step):
        img_=image_t[:,:,i:i+480,j:j+480]
        image_num[:,:,i:i+480,j:j+480]=image_num[:,:,i:i+480,j:j+480]+1
        with torch.no_grad():
            pred_d,pred_b =model(img_,flag_step2=True,x2_list=x2,kcam_list=kcam)
            pred_img[:,:,i:i+480,j:j+480]=pred_img[:,:,i:i+480,j:j+480]+pred_d
    print(i)
pred_img=pred_img.cpu()/image_num

plt.imshow(pred_img.detach().cpu().numpy()[0,0,:,:])
plt.show()

plt.imshow(img_.detach().cpu().numpy()[0,0,:,:])
plt.show()
        







#pad image with the new size
img_pad=torch.zeros((1,3,480*n_w,480*n_h))
img_pad[:,:,:image_t.shape[2],:image_t.shape[3]]=image_t

pred_img=torch.zeros(1,1,img_pad.shape[2],img_pad.shape[3])

img_tmp=torch.zeros_like(img_pad)

pred_list=[]
for i in range(n_w):
    for j in range(n_h):
        img_=img_pad[:,:,i*480:i*480+480,j*480:j*480+480].cuda()
        img_tmp[:,:,i*480:i*480+480,j*480:j*480+480]=img_
        with torch.no_grad():
            pred_d,pred_b =model(img_,flag_step2=True,x2_list=x2,kcam_list=kcam)
            pred_list.append(pred_d)
            pred_img[:,:,i*480:i*480+480,j*480:j*480+480]=pred_d
            





f=40e-3
base_f=25e-3
fdist=np.array([2])
kcam=(fdist-f)*(base_f**2)/(f**2)
x2=fdist.tolist()
kcam=kcam.tolist()

with torch.no_grad():
    pred_d,pred_b =model(image_t,flag_step2=True,x2_list=x2,kcam_list=kcam)


plt.imshow(image_t.detach().cpu().numpy()[0,0,:,:])
plt.show()
depth_gt[depth_gt>2]=0
plt.imshow(depth_gt.detach().cpu().numpy()[0,:,:])
plt.show()

plt.imshow(pred_img.detach().cpu().numpy()[0,0,:,:])
plt.show()

