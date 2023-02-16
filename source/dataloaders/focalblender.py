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
def get_blur(s1,s2,f):
    blur=abs(s2-s1)/s2 * 1/(s1-f*1e-3)
    return blur/10.

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

    def __init__(self, root_dir, transform_fnc=None,blur=1,aif=0,fstack=0,focus_dist=[0.1,.15,.3,0.7,1.5,100000],
                req_f_indx=[0,2], max_dpt = 3.):

        self.root_dir = root_dir
        print("image data root dir : " +str(self.root_dir))
        self.transform_fnc = transform_fnc

        self.blur=blur
        self.aif=aif
        self.fstack=fstack
        self.img_num = len(focus_dist)

        self.focus_dist = focus_dist
        self.req_f_idx=req_f_indx

        ##### Load and sort all images
        self.imglist_all = [f for f in listdir(root_dir) if isfile(join(root_dir, f)) and f[-7:] == "All.tif"]
        self.imglist_dpt = [f for f in listdir(root_dir) if isfile(join(root_dir, f)) and f[-7:] == "Dpt.exr"]
        self.imglist_allif = [f for f in listdir(root_dir) if isfile(join(root_dir, f)) and f[-7:] == "Aif.tif"]

        print("Total number of samples", len(self.imglist_dpt), "  Total number of seqs", len(self.imglist_dpt) / self.img_num)

        self.imglist_all.sort()
        self.imglist_dpt.sort()
        self.imglist_allif.sort()

        self.max_dpt = max_dpt

    def __len__(self):
        return int(len(self.imglist_dpt))

    def __getitem__(self, idx):
        if(self.fstack):
            #if stack is needed return all fdists from req_f_idx
            reqar=self.req_f_idx
        else:
            reqar=[random.choice(self.req_f_idx)]

        # add RGB, CoC, Depth inputs
        mats_input = np.zeros((256, 256, 3,0))
        mats_output = np.zeros((256, 256, 0))

        ##### Read and process an image
        idx_dpt = int(idx)
        img_dpt = read_dpt(self.root_dir + self.imglist_dpt[idx_dpt])
        #img_dpt_scaled = np.clip(img_dpt, 0., 1.9)
        #mat_dpt_scaled = img_dpt_scaled / 1.9
        mat_dpt_scaled = img_dpt/self.max_dpt
        mat_dpt = mat_dpt_scaled.copy()[:, :, np.newaxis]

        #extract N from the file name
        kcam=float(self.imglist_dpt[idx_dpt].split('_')[1])
        f=float(self.imglist_dpt[idx_dpt].split('_')[2])
       
        ind = idx * self.img_num

        #if all in-focus image is also needed append that to the input matrix
        if self.aif:
            im = Image.open(self.root_dir + self.imglist_allif[idx])
            img_all = np.array(im)
            mat_all = img_all.copy() / 255.
            mat_all=np.expand_dims(mat_all,axis=-1)
            mats_input = np.concatenate((mats_input, mat_all), axis=3)
        fdist=np.zeros((0))
        for req in reqar:
            im = Image.open(self.root_dir + self.imglist_all[ind + req])
            img_all = np.array(im)
            mat_all = img_all.copy() / 255.
            mat_all=np.expand_dims(mat_all,axis=-1)
            mats_input = np.concatenate((mats_input, mat_all), axis=3)

            img_msk = get_blur(self.focus_dist[req], img_dpt,f)
            #img_msk = np.clip(img_msk, 0, 1.0e-4) / 1.0e-4
            mat_msk = img_msk.copy()[:, :, np.newaxis]

            #append blur to the output
            mats_output = np.concatenate((mats_output, mat_msk), axis=2)
            fdist=np.concatenate((fdist,[self.focus_dist[req]]),axis=0)
        
        #append depth to the output
        mats_output = np.concatenate((mats_output, mat_dpt), axis=2)
        
        sample = {'input': mats_input, 'output': mats_output}

        if self.transform_fnc:
            sample = self.transform_fnc(sample)
        sample = {'input': sample['input'], 'output': sample['output'],'fdist':fdist,'kcam':kcam,'f':f*1e-3}
        return sample


class ToTensor(object):
    def __call__(self, sample):
        mats_input, mats_output = sample['input'], sample['output']

        mats_input = mats_input.transpose((3,2, 0, 1))
        mats_output = mats_output.transpose((2, 0, 1))
        return {'input': torch.from_numpy(mats_input),
                'output': torch.from_numpy(mats_output),}


def load_data(data_dir, blur,aif,train_split,fstack,
              WORKERS_NUM, BATCH_SIZE, FOCUS_DIST, REQ_F_IDX, MAX_DPT):
    img_dataset = ImageDataset(root_dir=data_dir,blur=blur,aif=aif,transform_fnc=transforms.Compose([ToTensor()]),
                               focus_dist=FOCUS_DIST,fstack=fstack,req_f_indx=REQ_F_IDX, max_dpt=MAX_DPT)

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


datapath='C:\\Users\\lahir\\focalstacks\\datasets\\mediumN1\\'
def get_data_stats(datapath):
    loaders, total_steps = load_data(datapath,blur=1,aif=0,train_split=0.8,fstack=0,WORKERS_NUM=0,
        BATCH_SIZE=1,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,100000],REQ_F_IDX=[0,1,2,3,4],MAX_DPT=1.0)
    print('stats of train data')
    get_loader_stats(loaders[0])
    print('______')

#data statistics of the input images
def get_loader_stats(loader):
    xmin,xmax,xmean,count=100,0,0,0
    depthmin,depthmax,depthmean=100,0,0
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


    print('X min='+str(xmin))
    print('X max='+str(xmax))
    print('X mean='+str(xmean/count))

    print('depth min='+str(depthmin))
    print('depth max='+str(depthmax))
    print('depth mean='+str(depthmean/count))
