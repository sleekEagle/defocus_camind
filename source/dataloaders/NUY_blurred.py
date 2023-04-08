import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, utils

from os import listdir, mkdir
from os.path import isfile, join, isdir
from visdom import Visdom
import numpy as np
import random
import OpenEXR
from PIL import Image
from skimage import img_as_float
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as F

CROP_PIX=15

#read kcams.txt file
def read_kcamfile(file):
    d = {}
    with open(file) as f:
        for line in f:
            if(len(line)<2):
                continue
            (key, val) = line.split()
            try:
                d[key] = float(val)
            except:
                 d[key] = val
    return d

# to calculate circle of confusion to train defocusNet
def _abs_val(x):
    if isinstance(x, np.ndarray) or isinstance(x, float) or isinstance(x, int):
        return np.abs(x)
    else:
        return x.abs()

class CameraLens:
    def __init__(self, focal_length, sensor_size_full=(0, 0), resolution=(1, 1), aperture_diameter=None, f_number=None, depth_scale=1):
        self.focal_length = focal_length
        self.depth_scale = depth_scale
        self.sensor_size_full = sensor_size_full

        if aperture_diameter is not None:
            self.aperture_diameter = aperture_diameter
            self.f_number = (focal_length / aperture_diameter) if aperture_diameter != 0 else 0
        else:
            self.f_number = f_number
            self.aperture_diameter = focal_length / f_number

        if self.sensor_size_full is not None:
            self.resolution = resolution
            self.aspect_ratio = resolution[0] / resolution[1]
            self.sensor_size = [self.sensor_size_full[0], self.sensor_size_full[0] / self.aspect_ratio]
        else:
            self.resolution = None
            self.aspect_ratio = None
            self.sensor_size = None
            self.fov = None
            self.focal_length_pixel = None

    def _get_indep_fac(self, focus_distance):
        return (self.aperture_diameter * self.focal_length) / (focus_distance - self.focal_length)

    def get_coc(self, focus_distance, depth):
        if isinstance(focus_distance, torch.Tensor):
            for _ in range(len(depth.shape) - len(focus_distance.shape)):
                focus_distance = focus_distance.unsqueeze(-1)

        return (_abs_val(depth - focus_distance) / depth) * self._get_indep_fac(focus_distance)

# reading depth files
def read_dpt(img_dpt_path): 
    dpt_img = OpenEXR.InputFile(img_dpt_path)
    dw = dpt_img.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    (r, g, b) = dpt_img.channels("RGB")
    dpt = np.fromstring(r, dtype=np.float16)
    dpt.shape = (size[1], size[0])
    return dpt

# to calculate circle of confusion
'''
output |s2-s1|/s2
'''
def get_blur(s1,s2,f,kcam):
    blur=abs(s2-s1)/s2 * 1/(s1-f*1e-3)*1/kcam
    return blur

'''
All in-focus image is attached to the input matrix after the RGB image

focus_dist - available focal dists in the dataset
req_f_indx - a list of focal dists we require.

fstack=0
a single image from the focal stack will be returned.
The image will be randomly selected from indices in req_f_indx
fstack=1
several images comprising of the focal stack will be returned. 
indices of the focal distances will be selected from req_f_indx

if aif=1 : 
    all in focus image will be appended at the begining of the focal stack

input matrix channles :
[batch,image,rgb_channel,256,256]

if aif=1 and fstack=1
image 0 : all in focus image
image 1:-1 : focal stack
if aif=0
image 0:-1 : focal stack

if fstack=0

output: [depth, blur1, blur2,]
blur1,blur2... corresponds to the focal stack
'''

class ImageDataset(torch.utils.data.Dataset):
    """Focal place dataset."""

    def __init__(self, rgbpath,depthpath, transform_fnc=None,blur=1,aif=0,fstack=0,focus_dist=[0.1,.15,.3,0.7,1.5,100000],
                req_f_indx=[0,2], max_dpt = 3.,blurclip=10.,kcampath=None):

        self.rgbpath=rgbpath
        self.depthpath=depthpath

        self.transform_fnc = transform_fnc

        self.blur=blur
        self.aif=aif
        self.fstack=fstack

        self.focus_dist = focus_dist
        self.req_f_idx=req_f_indx
        self.blurclip=blurclip
        self.kcampath=kcampath
        if(kcampath):
            self.kcamdict=read_kcamfile(kcampath)
            #calculate kcam based on parameters in file
            N=self.kcamdict['N']
            self.f=self.kcamdict['f']
            px=self.kcamdict['px']
            self.s1=self.kcamdict['focus']
            self.kcam=1/(self.f**2/N/px)
            print('kcam:'+str(self.kcam))
            print('f:'+str(self.f))

        ##### Load and sort all images
        self.imglist_rgb = [f for f in listdir(rgbpath) if isfile(join(rgbpath, f)) and f[-7:] == "rgb.png"]
        self.imglist_dpt = [f for f in listdir(depthpath) if isfile(join(depthpath, f)) and f[-4:] == ".png"]

        print("Total number of samples", len(self.imglist_dpt), "  Total number of seqs", len(self.imglist_dpt))

        self.imglist_rgb.sort()
        self.imglist_dpt.sort()

        self.max_dpt = max_dpt

    def __len__(self):
        return int(len(self.imglist_dpt))

    def __getitem__(self, idx):
        fdist=np.zeros((0))
        # add RGB, CoC, Depth inputs
        mats_input = np.zeros((256, 256, 3,0))
        mats_output = np.zeros((256, 256, 0))

        ##### Read and process an image
        #read depth image
        img_dpt = cv2.imread(self.depthpath + self.imglist_dpt[idx],cv2.IMREAD_UNCHANGED)
        #img_dpt_scaled = np.clip(img_dpt, 0., 1.9)
        #mat_dpt_scaled = img_dpt_scaled / 1.9
        mat_dpt_scaled = img_dpt/self.max_dpt
        mat_dpt = mat_dpt_scaled.copy()[:, :, np.newaxis]
        print('depth:'+str(mat_dpt.shape))

        #read rgb image
        im = cv2.imread(self.rgbpath + self.imglist_rgb[idx],cv2.IMREAD_UNCHANGED)
        img_rgb = np.array(im)
        mat_rgb = img_rgb.copy() / 255.
        print('rgb:'+str(mat_rgb.shape))
            
        img_blur = get_blur(self.s1, img_dpt,self.f,self.kcam)
        img_blur = img_blur / self.blurclip
        #img_msk = np.clip(img_msk, 0, 1.0e-4) / 1.0e-4
        mat_blur = img_blur.copy()[:, :, np.newaxis]
        print('blur:'+str(mat_blur.shape))

        data=np.concatenate((mat_rgb,mat_dpt,mat_blur),axis=2)
        data=torch.from_numpy(data)
        print('data:'+str(data.shape))

        fdist=np.concatenate((fdist,[self.s1]),axis=0)
                
        if self.transform_fnc:
            data_tr = self.transform_fnc(data)
            print('transformed:'+str(data_tr.shape))
        sample = {'rgb': data_tr[:3,:,:], 'depth': data_tr[3,:,:],'blur':data_tr[4,:,:],'fdist':fdist,'kcam':self.kcam,'f':self.f}
        return sample


class Transform(object):
    def __call__(self, image):
        image=torch.permute(image,(2,0,1))
        _,w,h=image.shape
        cropped=F.crop(image,CROP_PIX,CROP_PIX,w-CROP_PIX,h-CROP_PIX)
        return cropped
    


def load_data(rgbpath,depthpath, blur,aif,train_split,fstack,
              WORKERS_NUM, BATCH_SIZE, FOCUS_DIST, REQ_F_IDX, MAX_DPT,blurclip=10.0,kcampath=None):
    tr=transforms.Compose([
        Transform(),
        transforms.RandomCrop((256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
        ])
    img_dataset = ImageDataset(rgbpath=rgbpath,depthpath=depthpath,blur=blur,aif=aif,transform_fnc=tr,
                               focus_dist=FOCUS_DIST,fstack=fstack,req_f_indx=REQ_F_IDX, max_dpt=MAX_DPT,
                               blurclip=blurclip,kcampath=kcampath)

    indices = list(range(len(img_dataset)))
    split = int(len(img_dataset) * train_split)

    indices_train = indices[:split]
    indices_valid = indices[split:]

    dataset_train = torch.utils.data.Subset(img_dataset, indices_train)
    dataset_valid = torch.utils.data.Subset(img_dataset, indices_valid)

    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=WORKERS_NUM, batch_size=BATCH_SIZE, shuffle=True)
    loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid, num_workers=1, batch_size=1, shuffle=False)

    total_steps = int(len(dataset_train) / BATCH_SIZE)
    print("Total number of steps per epoch:", total_steps)
    print("Total number of training sample:", len(dataset_train))
    print("Total number of validataion sample:", len(dataset_valid))

    return [loader_train, loader_valid], total_steps


depthpath='C:\\Users\\lahir\\data\\nuy_depth\\depth\\'
rgbpath='C:\\Users\\lahir\\data\\nuy_depth\\refocused1\\'
kcampath='C:\\Users\\lahir\\data\\nuy_depth\\refocused1\\camparam.txt'
blurclip=1
loaders, total_steps = load_data(rgbpath=rgbpath,depthpath=depthpath,blur=1,aif=0,train_split=0.8,fstack=0,WORKERS_NUM=0,
        BATCH_SIZE=1,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,100000],REQ_F_IDX=[0,1,2,3,4],MAX_DPT=1.0,blurclip=blurclip,kcampath=kcampath)
for st_iter, sample_batch in enumerate(loaders[0]):
    rgb=sample_batch['rgb']
    depth=sample_batch['depth']
    blur=sample_batch['blur']
    break

import matplotlib.pyplot as plt
img=rgb[0,:,:,:]
img=torch.permute(img,(1,2,0))

d=torch.permute(depth,(1,2,0))

f, axarr = plt.subplots(1,2)
axarr[0].imshow(img)
axarr[1].imshow(d)


plt.imshow(img)

plt.imshow(d)
plt.show()


def get_data_stats(datapath,blurclip):
    loaders, total_steps = load_data(datapath,blur=1,aif=0,train_split=0.8,fstack=0,WORKERS_NUM=0,
        BATCH_SIZE=1,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,100000],REQ_F_IDX=[0,1,2,3,4],MAX_DPT=1.0,blurclip=blurclip,kcampath=kcampath)
    print('stats of train data')
    get_loader_stats(loaders[0])
    print('______')

#data statistics of the input images
def get_loader_stats(loader):
    xmin,xmax,xmean,count=100,0,0,0
    depthmin,depthmax,depthmean=100,0,0
    blurmin,blurmax,blurmean=100,0,0
    for st_iter, sample_batch in enumerate(loader):
        # Setting up input and output data
        X = sample_batch['input'][:,0,:,:,:].float()
        Y = sample_batch['output'].float()

        xmin_=torch.min(X).cpu().item()
        if(xmin_<xmin):
            xmin=xmin_
        xmax_=torch.max(X).cpu().item()
        if(xmax_>xmax):
            xmax=xmax_
        xmean+=torch.mean(X).cpu().item()
        count+=1
    
        #blur (|s2-s1|/(s2*(s1-f)))
        gt_step1 = Y[:, :-1, :, :]
        #depth in m
        gt_step2 = Y[:, -1:, :, :]
        
        depthmin_=torch.min(gt_step2).cpu().item()
        if(depthmin_<depthmin):
            depthmin=depthmin_
        depthmax_=torch.max(gt_step2).cpu().item()
        if(depthmax_>depthmax):
            depthmax=depthmax_
        depthmean+=torch.mean(gt_step2).cpu().item()

        blurmin_=torch.min(gt_step1).cpu().item()
        if(blurmin_<blurmin):
            blurmin=blurmin_
        blurmax_=torch.max(gt_step1).cpu().item()
        if(blurmax_>blurmax):
            blurmax=blurmax_
        blurmean+=torch.mean(gt_step1).cpu().item()

    print('X min='+str(xmin))
    print('X max='+str(xmax))
    print('X mean='+str(xmean/count))

    print('depth min='+str(depthmin))
    print('depth max='+str(depthmax))
    print('depth mean='+str(depthmean/count))

    print('blur min='+str(blurmin))
    print('blur max='+str(blurmax))
    print('blur mean='+str(blurmean/count))

'''
blur_thres=7.0
p=3.1/256*1e-3 # pixel width in m
N=2
f=6e-3
s2range=[0.1,2.0]
s1range=[0.1,1.5]

get_workable_s1s2ranges(p,N,f,s2range,s1range,blur_thres)
'''







