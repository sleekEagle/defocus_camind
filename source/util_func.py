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
import sys

def weights_init(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal(m.weight)
        m.bias.data.fill_(0.01)


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

def load_model(TRAIN_PARAMS):
    arch = importlib.import_module('arch.dofNet_arch' + str(TRAIN_PARAMS['ARCH_NUM']))

    ch_inp_num = 3
    ch_out_num = 1

    total_ch_inp = ch_inp_num
    model = arch.AENet(total_ch_inp, 1, TRAIN_PARAMS['FILTER_NUM'], flag_step2=True)
    model.apply(weights_init)

    params = list(model.parameters())
    print("model.parameters()", len(params))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable params/Total number:",
          str(pytorch_total_params_train) + "/" + str(pytorch_total_params))

    return model, ch_inp_num, ch_out_num


def set_comp_device(FLAG_GPU):
    device_comp = torch.device("cpu")
    if FLAG_GPU:
        device_comp = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device_comp


def set_output_folders(OUTPUT_PARAMS, TRAIN_PARAMS):
    model_name = 'a' + str(TRAIN_PARAMS['ARCH_NUM']).zfill(2) + '_exp' + str(
        OUTPUT_PARAMS['EXP_NUM']).zfill(2)
    models_dir = OUTPUT_PARAMS['MODEL_PATH'] + model_name + '/'
    if not isdir(models_dir):
        mkdir(models_dir)
    return models_dir, model_name

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



def forward_pass(X, model_info,stacknum=1,flag_step2=True,additional_input=0,foc_dist=0):
    outputs = model_info['model'](X, model_info['inp_ch_num'], stacknum, flag_step2=flag_step2, x2 = additional_input,foc_dist=foc_dist)
    if flag_step2:
        return (outputs[1], outputs[0])
    else:
        return outputs

def eval(loader,model_info,depthscale,fscale,s2limits,dataset=None,kcam=0,f=0,alt_gt=None):
    means2mse1,means2mse2,meanblurmse,meanblur=0,0,0,0
    print('Total samples = '+str(len(loader)))
    for st_iter, sample_batch in enumerate(loader):
        #sys.stdout.write(str(st_iter)+" of "+str(len(loader))+" is done")
        sys.stdout.write("\r%d is done"%st_iter)
        sys.stdout.flush()

        if(dataset=='ddff'):
            img_stack, gt_disp, foc_dist=sample_batch
            X=img_stack.float().to(model_info['device_comp'])
            Y=gt_disp.float().to(model_info['device_comp'])
            gt_step2=Y
        if(dataset=='blender'):
            X = sample_batch['input'][:,0,:,:,:].float().to(model_info['device_comp'])
            Y = sample_batch['output'].float().to(model_info['device_comp'])
            gt_step1 = Y[:, :-1, :, :]
            gt_step2 = Y[:, -1:, :, :]

        stacknum = 1

        if(len(s2limits)==2):
            mask=(gt_step2>s2limits[0]).int()*(gt_step2<s2limits[1]).int()
            s=torch.sum(mask).item()
            #continue loop if there are no ground truth data in the range we are interested in
            if(s==0):
                continue
        else:
            mask=torch.ones_like(gt_step2)
        
        X2_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
        s1_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
        for t in range(stacknum):
            #iterate through the batch
            for i in range(X.shape[0]):
                if(dataset=='blender'):
                    focus_distance=sample_batch['fdist'][i].item()
                    X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*(focus_distance-sample_batch['f'][i].item())/fscale*sample_batch['kcam'][i].item()/1.4398
                elif(dataset=='ddff'):
                    focus_distance=foc_dist[i].item()
                    X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*(focus_distance-f)/fscale*kcam/1.4398
                s1_fcs[i, t:(t + 1), :, :] = s1_fcs[i, t:(t + 1), :, :]*(focus_distance)/fscale
        X2_fcs = X2_fcs.float().to(model_info['device_comp'])
        s1_fcs = s1_fcs.float().to(model_info['device_comp'])
        output_step1,output_step2 = forward_pass(X, model_info,stacknum=stacknum, additional_input=X2_fcs,foc_dist=s1_fcs)

        #output_step1=output_step1*(0.1-2.9e-3)*7
        blurpred=output_step1
        #calculate s2 provided that s2>s1
        s2est=0.1*1./(1-blurpred)
        #blur mse
        if(dataset=='blender'):
            blurmse=torch.sum(torch.square(output_step1-gt_step1)*mask).item()/torch.sum(mask).item()
            meanblurmse+=blurmse
        #calculate MSE value
        mse1=torch.sum(torch.square(s2est-gt_step2)*mask).item()/torch.sum(mask).item()
        #mse_val, ssim_val, psnr_val=util_func.compute_all_metrics(output_step2*mask,gt_step2*mask)
        means2mse1+=mse1
        if(alt_gt is None):
            mse2=torch.sum(torch.square(output_step2*depthscale-gt_step2)*mask).item()/torch.sum(mask).item()
        else:
            mse2=torch.sum(torch.square(output_step2*depthscale-alt_gt)*mask).item()/torch.sum(mask).item()

        means2mse2+=mse2
    
        blur=torch.mean(output_step1).item()
        meanblur+=blur
        
    return means2mse1/len(loader),means2mse2/len(loader),meanblurmse/len(loader),meanblur/len(loader)

def kcamwise_blur(loader,model_info,depthscale,fscale,s2limits):
    means2mse1,means2mse2,meanblurmse,meanblur,meanblur_corrected=0,0,0,0,0
    kcams_all,meanblur_all,meanblur_corrected_all,mse_all=torch.empty(0),torch.empty(0),torch.empty(0),torch.empty(0)
    for st_iter, sample_batch in enumerate(loader):
        X = sample_batch['input'][:,0,:,:,:].float().to(model_info['device_comp'])
        Y = sample_batch['output'].float().to(model_info['device_comp'])
        gt_step1 = Y[:, :-1, :, :]
        gt_step2 = Y[:, -1:, :, :]
        stacknum = 1

        if(len(s2limits)==2):
            mask=(gt_step2>s2limits[0]).int()*(gt_step2<s2limits[1]).int()
            s=torch.sum(mask).item()
            if(s==0):
                continue
        else:
            mask=torch.ones_like(gt_step2)

        kcams=sample_batch['kcam']
        kcams_all=torch.cat((kcams_all,kcams))

        X2_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
        s1_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
        for t in range(stacknum):
            #iterate through the batch
            for i in range(X.shape[0]):
                focus_distance=sample_batch['fdist'][i].item()
                X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*(focus_distance-sample_batch['f'][i].item())/fscale*sample_batch['kcam'][i].item()/1.4398
                s1_fcs[i, t:(t + 1), :, :] = s1_fcs[i, t:(t + 1), :, :]*(focus_distance)/fscale
        X2_fcs = X2_fcs.float().to(model_info['device_comp'])
        s1_fcs = s1_fcs.float().to(model_info['device_comp'])

        output_step1,output_step2 = forward_pass(X, model_info,stacknum=stacknum, additional_input=X2_fcs,foc_dist=s1_fcs)

        meanblur=torch.mean(output_step1*mask,dim=2).mean(dim=2)[:,0].detach().cpu()
        meanblur_corrected=torch.mean(output_step1*X2_fcs*fscale*mask,dim=2).mean(dim=2)[:,0].detach().cpu()
        meanblur_corrected_all=torch.cat((meanblur_corrected_all,meanblur_corrected))

        meanblur_all=torch.cat((meanblur_all,meanblur))
        mse=torch.sum(torch.square((output_step2*depthscale-gt_step2)*mask),dim=2).sum(dim=2)[:,0].detach().cpu()/torch.sum(mask,dim=2).sum(dim=2)[:,0].detach().cpu()
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

    blurres_corrected = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, meanblur_corrected_all)
    blurres_corrected = blurres_corrected / labels_count

    mseres = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, mse_all)
    mseres = mseres / labels_count

    print(unique_kcams)
    print(blurres)
    print(mseres)

    #plot
    unique_kcams=unique_kcams.numpy()
    mseres=mseres.numpy()
    blurres=blurres.numpy()
    blurres_corrected=blurres_corrected.numpy()

    plt.scatter(unique_kcams,blurres)
    plt.scatter(unique_kcams,blurres_corrected)
    plt.title('Blur vs Kcam')
    plt.xlabel('Kcam')
    plt.ylabel('Blur')
    plt.show()

    plt.scatter(unique_kcams,blurres_corrected)
    plt.title('corrected Blur vs Kcam')
    plt.xlabel('Kcam')
    plt.ylabel('Corrected Blur')
    plt.show()

    plt.scatter(unique_kcams,mseres)
    plt.title('MSE vs Kcam')
    plt.xlabel('Kcam')
    plt.ylabel('MSE')
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
'''
import matplotlib.pyplot as plt
import numpy as np

def_mse=np.array([0.0500, 0.0427, 0.0477, 0.0547, 0.0737, 0.0723, 0.0948, 0.0746, 0.0870,0.0942, 0.0833, 0.1003])
our_mse=np.array([0.0582, 0.0586, 0.0583, 0.0528, 0.0528, 0.0478, 0.0592, 0.0567, 0.0521,0.0680, 0.0543, 0.0624])
f_numbers=np.array([1.0000, 1.1000, 1.2000, 1.5000, 1.8000, 2.0000, 2.1999, 2.4000, 2.5000,2.6000, 2.8000, 2.9999])

l1=plt.scatter(f_numbers,def_mse)
l2=plt.scatter(f_numbers,our_mse)
plt.title('MSE vs f-number')
plt.xlabel('f-number')
plt.ylabel('MSE m^2')
l1.set_label('defocusnet')
l2.set_label('our model')
plt.legend()
plt.show()
'''







