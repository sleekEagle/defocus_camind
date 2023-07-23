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
def get_blur(s1,s2,f):
    blur=(torch.abs(s2-s1)/s2/(s1-f)*f**2)*2000.0
    return blur

#selected_dirs: what rgb directories are being selected : a list of indices of sorted dir names
class nyudepthv2(BaseDataset):
    def __init__(self, data_path, rgb_dir,depth_dir,
                 is_train=True,is_blur=False, crop_size=(448, 576), scale_size=None):
        super().__init__(crop_size)

        print('crop_size:'+str(crop_size))
        if crop_size[0] > 480:
            scale_size = (int(crop_size[0]*640/480), crop_size[0])

        self.scale_size = scale_size
        self.is_train = is_train
        self.is_blur=is_blur
        self.data_path = os.path.join(data_path, 'nyu_depth_v2')
        self.rgbpath=os.path.join(self.data_path,rgb_dir)
        self.depthpath=os.path.join(self.data_path,depth_dir)
        rgb_dir='refocused_f_25_fdist_2'
        self.fdist=float(rgb_dir.split('_')[-1])
        self.f=float(rgb_dir.split('_')[2])*1e-3
        print('fdist:'+str(self.fdist))
        print('f:'+str(self.f))
        
        #read scene names
        scene_path=os.path.join(self.data_path, 'scenes.mat')
        self.scenes=scipy.io.loadmat(scene_path)['scenes']

        #read splits
        splits_path=os.path.join(self.data_path, 'splits.mat')
        splits=scipy.io.loadmat(splits_path)
        if is_train:
            self.file_idx=list(splits['trainNdxs'][:,0])
        else:
            self.file_idx=list(splits['testNdxs'][:,0])

        self.image_path_list = []
        self.depth_path_list = []

        with open('nyu_class_list.json', 'r') as f:
            self.class_list = json.load(f)
 
        phase = 'train' if is_train else 'test'
        print("Dataset: NYU Depth V2")
        print("# of %s images: %d" % (phase, len(self.file_idx)))

    def __len__(self):
        return len(self.file_idx)

    def __getitem__(self, idx):

        num=self.file_idx[idx]
        gt_path=os.path.join(self.depthpath,(str(num)+".png"))
        img_path=os.path.join(self.rgbpath,(str(num)+".png"))
        scene_name=self.scenes[num-1][0][0][:-5]

        class_id = -1
        for i, name in enumerate(self.class_list):
            if name in scene_name:
                class_id = i
                break

        assert class_id >= 0
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')

        if self.scale_size:
            image = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))
            depth = cv2.resize(depth, (self.scale_size[0], self.scale_size[1]))
        
        if self.is_train:
            if self.is_blur==1:
                image,depth = self.augment_training_data_blur(image, depth)
            else:
                image,depth = self.augment_training_data(image, depth)
        else:
            image,depth = self.augment_test_data(image, depth)

        depth = depth / 1000.0  # convert in meters
        blur=get_blur(self.fdist,depth,self.f)
        return {'image': image, 'depth': depth, 'blur':blur, 'class_id': class_id,'fdist':self.fdist,'f':self.f}

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

def get_loader_stats(loader):
    print('getting NUY v2 stats...')
    xmin,xmax,xmean,count=100,0,0,0
    depthmin,depthmax,depthmean=100,0,0
    blurmin,blurmax,blurmean=100,0,0
    depthlist=torch.empty(0)
    for st_iter, sample_batch in enumerate(loader):
        input_RGB = sample_batch['image']
        depth_gt = sample_batch['depth']
        class_id = sample_batch['class_id']
        gt_blur = sample_batch['blur']

        xmin_=torch.min(input_RGB).cpu().item()
        if(xmin_<xmin):
            xmin=xmin_
        xmax_=torch.max(input_RGB).cpu().item()
        if(xmax_>xmax):
            xmax=xmax_
        xmean+=torch.mean(input_RGB).cpu().item()
        count+=1
        mask=depth_gt>0
        depth_gt=depth_gt[mask]
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
    return depthlist


# data_path='D:\\data\\'
# rgb_dir='refocused_f_25_fdist_2'
# depth_dir='rawDepth'
# is_blur=True
# crop_size=(480,480)

# train_dataset=nyudepthv2(data_path=data_path,rgb_dir=rgb_dir,depth_dir=depth_dir,crop_size=crop_size,is_blur=is_blur,is_train=True)
# val_dataset=nyudepthv2(data_path=data_path,rgb_dir=rgb_dir,depth_dir=depth_dir,crop_size=crop_size,is_blur=is_blur,is_train=False)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
#                                            num_workers=0,pin_memory=True)

# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
#                                          num_workers=0,pin_memory=True)

# get_loader_stats(train_loader)


# from configs.train_options import TrainOptions
# from dataset.base_dataset import get_dataset
# import torch

# opt = TrainOptions()
# args = opt.initialize().parse_args()
# args.shift_window_test=True
# args.flip_test=True

# dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path,'rgb_dir':args.rgb_dir, 'depth_dir':args.depth_dir,
#                   'selected_dirs':args.selected_dirs}
# dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

# train_dataset = get_dataset(**dataset_kwargs,is_train=True)
# # val_dataset = get_dataset(**dataset_kwargs, is_train=False)


# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,pin_memory=True)
# # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,pin_memory=True)
# loader=train_loader

# get_loader_stats(loader)
