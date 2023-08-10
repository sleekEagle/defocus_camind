# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import os
import cv2
from dataloaders.base_dataset import BaseDataset
import json
import scipy
import torch
import numpy as np
import random

#f in mm
base_f=25e-3
def get_blur(s1,s2,f):
    blur=torch.abs(s2-s1)/s2/(s1-f)*(f**2)/(base_f**2)
    return blur

#selected_dirs: what rgb directories are being selected : a list of indices of sorted dir names
class nyudepthv2(BaseDataset):
    def __init__(self, data_path,is_train=True,is_blur=False, crop_size=(480, 480)):
        super().__init__(crop_size)

        print('crop_size:'+str(crop_size))
    
        self.is_train = is_train
        self.is_blur=is_blur
        self.depthpath = os.path.join(data_path, 'mobiledepth')
        self.rgbpath=os.path.join(data_path,'mobilergb')
        img_nums=[int(name[:-4]) for name in os.listdir(self.rgbpath)]
        if is_train:
            self.file_idx=[n for n in img_nums if n<800]
        else:
            self.file_idx=[n for n in img_nums if n>=800]

        phase = 'train' if is_train else 'test'
        print("Dataset: NYU Depth V2")
        print("# of %s images: %d" % (phase, len(self.file_idx)))

    def __len__(self):
        return len(self.file_idx)

    def __getitem__(self, idx):

        num=self.file_idx[idx]
        
        rgbpath=os.path.join(self.rgbpath,str(num)+'.jpg')
        depthpath=os.path.join(self.depthpath,(str(num)+".png"))
        
        image = cv2.imread(rgbpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED).astype('float32')

        if self.is_train:
            if self.is_blur==1:
                image,depth = self.augment_training_data_blur(image, depth)
            else:
                image,depth = self.augment_training_data(image, depth)
        else:
            image,depth = self.augment_test_data(image, depth)

        depth = depth / 1000.0  # convert in meters
        blur=get_blur(2.0,depth,25e-3)
        return {'image': image, 'depth': depth, 'blur':blur}

# for st_iter, sample_batch in enumerate(loader):
#         input_RGB = sample_batch['image']
#         depth_gt = sample_batch['depth']
#         class_id = sample_batch['class_id']
#         gt_blur = sample_batch['blur']
#         break

# import matplotlib.pyplot as plt 
# gt_blur[gt_blur==-1]=0
# b=(gt_blur.numpy())[0,:,:]
# plt.imshow(b)
# plt.show()

# d=(depth_gt.numpy())[0,:,:]
# plt.imshow(d)
# plt.show()

def get_loader_stats(loader,depthrange=2.0):
    print('getting mobilekinect v2 stats...')
    xmin,xmax,xmean,count=100,0,0,0
    depthmin,depthmax,depthmean=100,0,0
    blurmin,blurmax,blurmean=100,0,0
    depthlist=torch.empty(0)
    for st_iter, sample_batch in enumerate(loader):
        input_RGB = sample_batch['image']
        depth_gt = sample_batch['depth']
        gt_blur = sample_batch['blur']

        xmin_=torch.min(input_RGB).cpu().item()
        if(xmin_<xmin):
            xmin=xmin_
        xmax_=torch.max(input_RGB).cpu().item()
        if(xmax_>xmax):
            xmax=xmax_
        xmean+=torch.mean(input_RGB).cpu().item()
        mask=(depth_gt>0)*(depth_gt<depthrange)
        depth_gt=depth_gt[mask]
        if depth_gt.shape[0]<100:
            continue
        count+=1
        t=torch.flatten(depth_gt)
        depthlist=torch.concat((depthlist,t),axis=0)
        depthmin_=torch.min(depth_gt).cpu().item()
        if(depthmin_<depthmin):
            depthmin=depthmin_
        depthmax_=torch.max(depth_gt).cpu().item()
        if(depthmax_>depthmax):
            depthmax=depthmax_
        depthmean+=torch.mean(depth_gt).cpu().item()
        gt_blur=gt_blur[mask]
        blurmin_=torch.min(gt_blur).cpu().item()
        if(blurmin_<blurmin):
            blurmin=blurmin_
        blurmax_=torch.max(gt_blur).cpu().item()
        if(blurmax_>blurmax):
            blurmax=blurmax_
        blurmean+=torch.mean(gt_blur).cpu().item()

    print('RGB min='+str(xmin))
    print('RGB max='+str(xmax))
    print('RGB mean='+str(xmean/count))

    print('depth min='+str(depthmin))
    print('depth max='+str(depthmax))
    print('depth mean='+str(depthmean/count))

    print('blur min='+str(blurmin))
    print('blur max='+str(blurmax))
    print('blur mean='+str(blurmean/count))
    return depthlist,count


# data_path='C:\\Users\\lahir\\data\\kinectmobile\\'
# train_dataset=nyudepthv2(data_path=data_path,is_train=True)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
#                                            num_workers=0,pin_memory=True)

# depthlist,count=get_loader_stats(train_loader,depthrange=10.0)
# #plot histogram of depths
# import matplotlib.pyplot as plt
# n, bins, patches=plt.hist(depthlist.numpy(),20)
# plt.show()

# for st_iter, sample_batch in enumerate(train_loader):
#     input_RGB = sample_batch['image']
#     depth_gt = sample_batch['depth']
#     gt_blur = sample_batch['blur']
#     break
# import matplotlib.pyplot as plt
# img=input_RGB[0,0,:,:]
# plt.imshow(img)
# plt.show()


