#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:54:12 2018
@author: maximov
"""

import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, utils

from os import listdir, mkdir
from os.path import isfile, join, isdir
from visdom import Visdom
import numpy as np
import importlib
import random
import csv
import OpenEXR, Imath
from PIL import Image
from skimage import img_as_float
import skimage
if(skimage.__version__ >= '0.18'):
    from skimage import metrics
else:
    from skimage import measure
from scipy import stats
import math
import matplotlib.pyplot as plt


def _abs_val(x):
    if isinstance(x, np.ndarray) or isinstance(x, float) or isinstance(x, int):
        return np.abs(x)
    else:
        return x.abs()

# reading depth files
def read_dpt(img_dpt_path):
    # pt = Imath.PixelType(Imath.PixelType.HALF)  # FLOAT HALF
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

        #append depth to the output
        mats_output = np.concatenate((mats_output, mat_dpt), axis=2)

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
            fdist=np.concatenate((fdist,[self.focus_dist[req]/self.max_dpt]),axis=0)
        
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


def weights_init(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal(m.weight)
        m.bias.data.fill_(0.01)

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

'''
data_dir='C:\\Users\\lahir\\focalstacks\\datasets\\mediumN1\\'
loaders, total_steps = load_data(data_dir,blur=1,aif=1,train_split=0.8,fstack=1,WORKERS_NUM=0,
BATCH_SIZE=10,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,100000],REQ_F_IDX=[0],MAX_DPT=3.)


for st_iter, sample_batch in enumerate(loaders[0]):
    break

import matplotlib.pyplot as plt
i=0
for j in range(0,8):
    img=sample_batch['input'][i,j,:,:,:].transpose(0,-1).detach().cpu().numpy()
    plt.imshow(img)
    plt.show()
'''

def load_model(model_dir, model_name, TRAIN_PARAMS, DATA_PARAMS):
    arch = importlib.import_module('arch.dofNet_arch' + str(TRAIN_PARAMS['ARCH_NUM']))

    ch_inp_num = 0
    if DATA_PARAMS['FLAG_IO_DATA']['INP_RGB']:
        ch_inp_num += 3
    if DATA_PARAMS['FLAG_IO_DATA']['INP_COC']:
        ch_inp_num += 1

    ch_out_num = 0

    if DATA_PARAMS['FLAG_IO_DATA']['OUT_DEPTH']:
        ch_out_num += 1
    ch_out_num_all = ch_out_num
    if DATA_PARAMS['FLAG_IO_DATA']['OUT_COC']:
        ch_out_num_all = ch_out_num + 1 * DATA_PARAMS['INP_IMG_NUM']
        ch_out_num += 1

    total_ch_inp = ch_inp_num * DATA_PARAMS['INP_IMG_NUM']
    if TRAIN_PARAMS['ARCH_NUM'] > 0:
        total_ch_inp = ch_inp_num

        flag_step2 = False
        if TRAIN_PARAMS['TRAINING_MODE'] == 2:
            flag_step2 = True
        model = arch.AENet(total_ch_inp, 1, TRAIN_PARAMS['FILTER_NUM'], flag_step2=flag_step2)
    else:
        model = arch.AENet(total_ch_inp, ch_out_num_all, TRAIN_PARAMS['FILTER_NUM'])
    model.apply(weights_init)

    params = list(model.parameters())
    print("model.parameters()", len(params))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable params/Total number:",
          str(pytorch_total_params_train) + "/" + str(pytorch_total_params))

    if TRAIN_PARAMS['EPOCH_START'] > 0:
        model.load_state_dict(torch.load(model_dir + model_name + '_ep' + str(TRAIN_PARAMS['EPOCH_START']) + '.pth'))
        print("Model loaded:", model_name, " epoch:", str(TRAIN_PARAMS['EPOCH_START']))

    return model, ch_inp_num, ch_out_num


def set_comp_device(FLAG_GPU):
    device_comp = torch.device("cpu")
    if FLAG_GPU:
        device_comp = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device_comp


def set_output_folders(OUTPUT_PARAMS, DATA_PARAMS, TRAIN_PARAMS):
    model_name = 'a' + str(TRAIN_PARAMS['ARCH_NUM']).zfill(2) + '_d' + str(DATA_PARAMS['DATA_NUM']).zfill(2) + '_t' + str(
        OUTPUT_PARAMS['EXP_NUM']).zfill(2)
    res_dir = OUTPUT_PARAMS['RESULT_PATH'] + model_name + '/'
    models_dir = OUTPUT_PARAMS['MODEL_PATH'] + model_name + '/'
    if not isdir(models_dir):
        mkdir(models_dir)
    if not isdir(res_dir):
        mkdir(res_dir)
    return models_dir, model_name, res_dir


def compute_loss(Y_est, Y_gt, criterion):
    return criterion(Y_est, Y_gt)


def compute_psnr(img1, img2, mode_limit=False, msk=0):
    if mode_limit:
        msk_num = np.sum(msk)
        mse = np.sum(msk * ((img1 - img2) ** 2)) / msk_num
    else:
        mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def compute_ssim(mat_est, mat_gt, mode_limit=False, msk=0):
    if(skimage.__version__ >= '0.18'):
        ssim_full = metrics.structural_similarity((mat_gt), (mat_est), data_range=img_as_float(mat_gt).max() - img_as_float(mat_gt).min(), channel_axis=-1,
                     full=True)
    else:
        ssim_full = measure.compare_ssim((mat_gt), (mat_est), data_range=img_as_float(mat_gt).max() - img_as_float(mat_gt).min(), multichannel=True,
                     full=True)
    if mode_limit:
        ssim_mean = np.sum(ssim_full[1]*msk) / (np.sum(msk))
    else:
        ssim_mean = np.sum(ssim_full[1]) / (mat_gt.shape[0] * mat_gt.shape[1] * mat_gt.shape[2])
    # dssim_mean = (1. - ssim_mean) / 2.
    return ssim_mean


def compute_pearson(a, b, mode_limit=False):
    a, b = a.flat, b.flat
    if mode_limit:
        m = np.argwhere(b > (2. / 8.))
        a = np.delete(a, m)
        b = np.delete(b, m)
    if len(a) < 10:
        coef = 0
    else:
        coef, p = stats.pearsonr(a, b)
    return coef

def compute_all_metrics(est_out, gt_out, flag_mse=True, flag_ssim=True, flag_psnr=True, flag_pearson=False, mode_limit=False):
    mat_gt = (gt_out[0]).to(torch.device("cpu")).data.numpy().transpose((1, 2, 0))
    mat_est = (est_out[0]).to(torch.device("cpu")).data.numpy().transpose((1, 2, 0))
    mat_est = np.clip(mat_est, 0., 1.)
    mse_val, ssim_val, psnr_val = 1., 0., 0.
    msk = mat_gt < 0.2
    msk_num = np.sum(msk)

    if msk_num==0:
        if flag_pearson:
            return 0, 0, 0, 0
        else:
            return 0, 0, 0

    if flag_ssim:
        ssim_val = compute_ssim(mat_gt, mat_est, mode_limit=mode_limit, msk=msk)
    if flag_psnr:
        psnr_val = compute_psnr(mat_gt, mat_est, mode_limit=mode_limit, msk=msk)
    if flag_mse:
        if mode_limit:
            mse_val = np.sum(msk*((mat_gt - mat_est) ** 2))/msk_num
        else:
            mse_val = np.mean((mat_gt - mat_est) ** 2)
    if flag_pearson:
        pearson_val = compute_pearson(mat_est, mat_gt, mode_limit=mode_limit)
        return mse_val, ssim_val, psnr_val, pearson_val
    return mse_val, ssim_val, psnr_val



# Visualize current progress
class Visualization():
    def __init__(self, port, hostname, model_name, flag_show_input=False, flag_show_mid=False, env_name='main'):
        self.viz = Visdom(port=port, server=hostname, env=env_name)
        self.loss_plot = self.viz.line(X=[0.], Y=[0.], name="train", opts=dict(title='Loss ' + model_name))
        self.flag_show_input = flag_show_input
        self.flag_show_mid = flag_show_mid

    def initial_viz(self, loss_val, viz_out, viz_gt_img, viz_inp, viz_mid):
        self.viz.line(Y=[loss_val], X=[0], win=self.loss_plot, name="train", update='replace')

        viz_out_img = torch.clamp(viz_out, 0., 1.)
        if viz_out.shape[1] > 3 or viz_out.shape[1] == 2:
            viz_out_img = viz_out_img[:, 0:1, :, :]
            viz_gt_img = viz_gt_img[:, 0:1, :, :]

        if self.flag_show_mid:
            viz_mid_img = torch.clamp(viz_mid[0, :, :, :], 0., 1.)
            viz_mid_img = viz_mid_img.unsqueeze(1)
            self.img_mid = self.viz.images(viz_mid_img, nrow=8)
        if self.flag_show_input:
            viz_inp_img = viz_inp[:, 0:3, :, :]
            self.img_input = self.viz.images(viz_inp_img, nrow=8)

        self.img_fit = self.viz.images(viz_out_img, nrow=8)
        self.img_gt = self.viz.images(viz_gt_img, nrow=8)

    def log_viz_img(self, viz_out, viz_gt_img, viz_inp, viz_mid):
        viz_out_img = torch.clamp(viz_out, 0., 1.)

        if viz_out.shape[1] > 3 or viz_out.shape[1] == 2:
            viz_out_img = viz_out_img[:, 0:1, :, :]
            viz_gt_img = viz_gt_img[:, 0:1, :, :]

        if self.flag_show_mid:
            viz_mid_img = torch.clamp(viz_mid[0, :, :, :], 0., 1.)
            viz_mid_img = viz_mid_img.unsqueeze(1)
            self.viz.images(viz_mid_img, win=self.img_mid, nrow=8)

        if self.flag_show_input:
            viz_inp_img = viz_inp[:, 0:3, :, :]
            self.viz.images(viz_inp_img, win=self.img_input, nrow=8)

        self.viz.images(viz_out_img, win=self.img_fit, nrow=8)
        self.viz.images(viz_gt_img, win=self.img_gt, nrow=8)

    def log_viz_plot(self, loss_val, total_iter):
        self.viz.line(Y=[loss_val], X=[total_iter], win=self.loss_plot, name="train", update='append')


def save_config(r, postfix="single"):
    model_name = 'a' + str(r.config['TRAIN_PARAMS']['ARCH_NUM']) + '_d' + str(r.config['DATA_PARAMS']['DATA_NUM']) + '_t' + str(
        r.config['OUTPUT_PARAMS']['EXP_NUM']).zfill(2)
    with open(r.config['OUTPUT_PARAMS']['RESULT_PATH'] + 'configs_' + postfix + '.csv', mode='a') as res_file:
        res_writer = csv.writer(res_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        res_writer.writerow([model_name, r.config['TRAIN_PARAMS'], r.config['DATA_PARAMS'], r.config['OUTPUT_PARAMS']])



def forward_pass(X, model_info, TRAIN_PARAMS,DATA_PARAMS,stacknum=1, additional_input=None,foc_dist=0):
    flag_step2 = True if TRAIN_PARAMS['TRAINING_MODE']==2 else False
    outputs = model_info['model'](X, model_info['inp_ch_num'], stacknum, flag_step2=flag_step2, x2 = additional_input,foc_dist=foc_dist,parallel=False)
    return (outputs[1], outputs[0]) if TRAIN_PARAMS['TRAINING_MODE']==2 else (outputs, outputs)

def eval(loader,model_info,TRAIN_PARAMS,DATA_PARAMS):
    means2mse1,means2mse2,meanblurmse,meanblur=0,0,0,0
    for st_iter, sample_batch in enumerate(loader):
        X = sample_batch['input'].float().to(model_info['device_comp'])
        Y = sample_batch['output'].float().to(model_info['device_comp'])
        if TRAIN_PARAMS['TRAINING_MODE'] == 2:
            gt_step1 = Y[:, :-1, :, :]
            gt_step2 = Y[:, -1:, :, :]
        stacknum = DATA_PARAMS['INP_IMG_NUM']
        focus_dists = DATA_PARAMS['FOCUS_DIST']

        if(True):
            mask=(gt_step2>0.1).int()*(gt_step2<3.0).int()
        else:
            mask=torch.ones_like(gt_step2)
        
        X2_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
        s1_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
        for t in range(stacknum):
            #iterate through the batch
            for i in range(X.shape[0]):
                focus_distance=sample_batch['fdist'][i].item()
                #X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :] * (focus_distance-sample_batch['f'][i].item())*sample_batch['kcam'][i].item()
                X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*(focus_distance-sample_batch['f'][i].item())*sample_batch['kcam'][i].item()/1.4398
                s1_fcs[i, t:(t + 1), :, :] = s1_fcs[i, t:(t + 1), :, :]*(focus_distance)
        X2_fcs = X2_fcs.float().to(model_info['device_comp'])
        s1_fcs = s1_fcs.float().to(model_info['device_comp'])

        output_step1,output_step2 = forward_pass(X, model_info,TRAIN_PARAMS,DATA_PARAMS,stacknum=stacknum, additional_input=X2_fcs,foc_dist=s1_fcs)

        #output_step1=output_step1*(0.1-2.9e-3)*7
        blurpred=output_step1
        #calculate s2 provided that s2>s1
        s2est=0.1*1./(1-blurpred)
        #blur mse
        blurmse=torch.sum(torch.square(output_step1-gt_step1)*mask).item()/torch.sum(mask).item()
        meanblurmse+=blurmse
        #calculate MSE value
        mse1=torch.sum(torch.square(s2est-gt_step2)*mask).item()/torch.sum(mask).item()
        #mse_val, ssim_val, psnr_val=util_func.compute_all_metrics(output_step2*mask,gt_step2*mask)
        means2mse1+=mse1
        mse2=torch.sum(torch.square(output_step2-gt_step2)*mask).item()/torch.sum(mask).item()
        means2mse2+=mse2
    
        blur=torch.mean(output_step1).item()
        meanblur+=blur
        
    return means2mse1/len(loader),means2mse2/len(loader),meanblurmse/len(loader),meanblur/len(loader)

def kcamwise_blur(loader,model_info,TRAIN_PARAMS,DATA_PARAMS):
    means2mse1,means2mse2,meanblurmse,meanblur=0,0,0,0
    kcams_all,meanblur_all,mse_all=torch.empty(0),torch.empty(0),torch.empty(0)
    for st_iter, sample_batch in enumerate(loader):
        X = sample_batch['input'].float().to(model_info['device_comp'])
        Y = sample_batch['output'].float().to(model_info['device_comp'])
        if TRAIN_PARAMS['TRAINING_MODE'] == 2:
            gt_step1 = Y[:, :-1, :, :]
            gt_step2 = Y[:, -1:, :, :]
        stacknum = DATA_PARAMS['INP_IMG_NUM']
        focus_dists = DATA_PARAMS['FOCUS_DIST']

        kcams=sample_batch['kcam']
        kcams_all=torch.cat((kcams_all,kcams))


        if(True):
            mask=(gt_step2>0.1).int()*(gt_step2<3.0).int()
        else:
            mask=torch.ones_like(gt_step2)

        X2_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
        s1_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
        for t in range(stacknum):
            #iterate through the batch
            for i in range(X.shape[0]):
                focus_distance=sample_batch['fdist'][i].item()
                #X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :] * (focus_distance-sample_batch['f'][i].item())*sample_batch['kcam'][i].item()*10
                X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*(focus_distance-sample_batch['f'][i].item())*sample_batch['kcam'][i].item()/1.4398
                s1_fcs[i, t:(t + 1), :, :] = s1_fcs[i, t:(t + 1), :, :]*(focus_distance)
        X2_fcs = X2_fcs.float().to(model_info['device_comp'])
        s1_fcs = s1_fcs.float().to(model_info['device_comp'])

        output_step1,output_step2 = forward_pass(X, model_info,TRAIN_PARAMS,DATA_PARAMS,stacknum=stacknum, additional_input=X2_fcs,foc_dist=s1_fcs)

        meanblur=torch.mean(output_step1,dim=2).mean(dim=2)[:,0].detach().cpu()
        meanblur_all=torch.cat((meanblur_all,meanblur))
        mse=torch.sum(torch.square(output_step2-gt_step2),dim=2).sum(dim=2)[:,0].detach().cpu()/torch.sum(mask,dim=2).sum(dim=2)[:,0].detach().cpu()
        mse_all=torch.cat((mse_all,mse))

    labels=torch.zeros_like(kcams_all,dtype=torch.int64)
    #get kcam wise mean blur
    unique_kcams, _ = kcams_all.unique(dim=0, return_counts=True)
    for i in range(unique_kcams.shape[0]):
        indices=((kcams_all == unique_kcams[i].item()).nonzero(as_tuple=True)[0])
        labels[indices]=i
    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    
    blurres = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, meanblur_all)
    blurres = blurres / labels_count
    mseres = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, mse_all)
    mseres = mseres / labels_count

    print(unique_kcams)
    print(blurres)
    print(mseres)

    #plot
    unique_kcams=unique_kcams.numpy()
    mseres=mseres.numpy()
    blurres=blurres.numpy()

    plt.scatter(unique_kcams,blurres)
    plt.title('Blur')
    plt.show()

    plt.scatter(unique_kcams,mseres)
    plt.title('MSE')
    plt.show()

'''
import torch
import matplotlib.pyplot as plt

unique_kcams=torch.tensor([1.4399, 1.5839, 1.7279, 2.1598, 2.5918, 2.8798, 3.1677, 3.4557, 3.5997,
        3.7437, 4.0317, 4.3196], dtype=torch.float64)

res=torch.tensor([0.6605, 0.6040, 0.5515, 0.4596, 0.3568, 0.3073, 0.2747, 0.2672, 0.2312,
        0.2310, 0.2126, 0.1956])

res=res.numpy()
unique_kcams=unique_kcams.numpy()

plt.scatter(unique_kcams,res)
plt.show()

#calculate the actual blur that should be present at each camera based on the first blur
res_theoratical=res[0]*unique_kcams[0]/unique_kcams
res_corrected=res*unique_kcams


plt.scatter(unique_kcams,res_corrected)
plt.scatter(unique_kcams,res_theoratical)
plt.scatter(unique_kcams,res)
plt.show()
'''





