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
        OUTPUT_PARAMS['EXP_NAME']).zfill(2)
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



def forward_pass(X, model_info,stacknum=1,camind=True,flag_step2=True,camparam=0,foc_dist=0,aif=False):
    outputs = model_info['model'](X,model_info['inp_ch_num'],stacknum,camind=camind,flag_step2=flag_step2,
    camparam=camparam,foc_dist=foc_dist)
    if aif:
        return (outputs[1],outputs[0])
    if flag_step2:
        return (outputs[1], outputs[0],outputs[2])
    else:
        return outputs

def eval(loader,model_info,depthscale,fscale,s2limits,camind=True,dataset=None,kcam=0,f=0,aif=False,calc_distmse=False):
    means2mse1,means2mse2,meanblurmse,meanblur=0,0,0,0
    minblur,maxblur,gt_meanblur=100,0,0
    #store distance wise mse
    distmse,distsum,distblur=torch.zeros(100),torch.zeros(100),torch.zeros(100)

    print('Total samples = '+str(len(loader)))
    for st_iter, sample_batch in enumerate(loader):
        #if(st_iter>100):
        #    break
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
                    f=sample_batch['f'].item()
                    #X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*(focus_distance-sample_batch['f'][i].item())/fscale*sample_batch['kcam'][i].item()/1.4398
                    #X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]/sample_batch['kcam'][i].item()*1.4398*(focus_distance-sample_batch['f'][i].item())/fscale
                    if(not aif):
                        X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*sample_batch['kcam'][i].item()*(focus_distance-f)/fscale
                elif(dataset=='ddff'):
                    focus_distance=foc_dist[i].item()
                    if(not aif):
                        X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*kcam*(focus_distance-f)/fscale
                if(not aif):
                    s1_fcs[i, t:(t + 1), :, :] = s1_fcs[i, t:(t + 1), :, :]*(focus_distance)/fscale
        X2_fcs = X2_fcs.float().to(model_info['device_comp'])
        s1_fcs = s1_fcs.float().to(model_info['device_comp'])
        if(aif):
            output_step1,output_step2 = forward_pass(X,model_info,stacknum=stacknum,camind=camind,camparam=X2_fcs,foc_dist=s1_fcs,aif=aif)
        else:
            output_step1,output_step2,corrected_blur = forward_pass(X,model_info,stacknum=stacknum,camind=camind,camparam=X2_fcs,foc_dist=s1_fcs,aif=aif)

        #gt_blur=torch.cat((gt_blur,torch.flatten(gt_step1.detach().cpu())))
        #pred_blur=torch.cat((pred_blur,torch.flatten(output_step1.detach().cpu())))

        minblur_=torch.min(output_step1).item()
        if(minblur_<minblur):   
            minblur=minblur_
        maxblur_=torch.max(output_step1).item()
        if(maxblur_>maxblur):
            maxblur=maxblur_

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
        mse2=torch.sum(torch.square(output_step2*depthscale-gt_step2)*mask).item()/torch.sum(mask).item()
        means2mse2+=mse2

        if(calc_distmse):
            squareder=torch.square(output_step2*depthscale-gt_step2)
            gtround=torch.round(gt_step2*10,decimals=0)
            for i in range(1,len(distmse)+1):
                selected_val=squareder[gtround==i]
                selected_blur=corrected_blur[gtround==i]
                #mask_sum=torch.sum(mask[gtround==i]).item()
                er=(torch.mean(selected_val)).item()
                b=(torch.mean(selected_blur)).item()
                if(not(math.isnan(er) or math.isnan(b))):
                    distmse[i-1]+=er
                    distblur[i-1]+=b
                    distsum[i-1]+=1

        blur=torch.sum(output_step1*mask).item()/torch.sum(mask).item()
        meanblur+=blur
        if(dataset=='blender'):
            gtblur=torch.sum(gt_step1*mask).item()/torch.sum(mask).item()
            gt_meanblur+=gtblur
    if(calc_distmse):
        print('\ndistance wise error (distances rounded to the shown value): ')
        mse_=distmse/distsum
        blur_=distblur/distsum
        mse_=mse_[~torch.isnan(mse_)]
        blur_=blur_[~torch.isnan(blur_)]
        values=np.arange(0.1,(len(mse_)+1)*0.1,0.1)
        for i,v in enumerate(values):
            #print("dist: %2.1f m : MSE: %4.3f blur: %4.3f"%(v,mse_[i].item(),blur_[i].item()))
            print("%4.3f"%(mse_[i].item()),end=",")
    return means2mse1/len(loader),means2mse2/len(loader),meanblurmse/len(loader),meanblur/len(loader),gt_meanblur/len(loader),minblur,maxblur

def kcamwise_blur(loader,model_info,depthscale,fscale,s2limits,camind,aif):
    print('iscamind:'+str(camind))
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
                f=sample_batch['f'].item()
                #X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*(focus_distance-sample_batch['f'][i].item())/fscale*(sample_batch['kcam'][i].item())/1.4398 * 0.9**(sample_batch['kcam'][i].item())
                if(not aif):
                    X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*sample_batch['kcam'][i].item()*(focus_distance-f)/fscale
                #X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*(focus_distance-sample_batch['f'][i].item())/fscale*1
                #X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*sample_batch['kcam'][i].item()/1.4398*(0.9**(sample_batch['kcam'][i].item()))
                if(not aif):
                    s1_fcs[i, t:(t + 1), :, :] = s1_fcs[i, t:(t + 1), :, :]*(focus_distance)/fscale
        X2_fcs = X2_fcs.float().to(model_info['device_comp'])
        s1_fcs = s1_fcs.float().to(model_info['device_comp'])

        if(aif):
            output_step1,output_step2 = forward_pass(X,model_info,stacknum=stacknum,camind=camind,camparam=X2_fcs,foc_dist=s1_fcs,aif=aif)
            mul=output_step1
        else:
            output_step1,output_step2,mul = forward_pass(X,model_info,stacknum=stacknum,camind=camind,camparam=X2_fcs,foc_dist=s1_fcs,aif=aif)


        meanblur=torch.mean(output_step1*mask,dim=2).mean(dim=2)[:,0].detach().cpu()
        meanblur_corrected=torch.mean(mul*mask,dim=2).mean(dim=2)[:,0].detach().cpu()
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

    theoratical_blur=torch.tensor([blurres[0].item()]).repeat_interleave(repeats=blurres.shape[0])*unique_kcams[0].item()
    theoratical_blur=theoratical_blur/unique_kcams

    #plot
    unique_kcams=unique_kcams.numpy()
    mseres=mseres.numpy()
    blurres=blurres.numpy()
    blurres_corrected=blurres_corrected.numpy()

    line1=plt.scatter(unique_kcams,blurres)
    line2=plt.scatter(unique_kcams,blurres_corrected)
    line3=plt.scatter(unique_kcams,theoratical_blur)
    line1.set_label('Blur')
    line2.set_label('Corrected Blur')
    line3.set_label('Theoratically expected blur')
    plt.title('Blur vs Kcam')
    plt.xlabel('Kcam')
    plt.ylabel('Blur')
    plt.legend()
    plt.show()

    plt.scatter(unique_kcams,mseres)
    plt.title('MSE vs Kcam')
    plt.xlabel('Kcam')
    plt.ylabel('MSE')
    plt.show()


'''
get the ranges of s1 and s2 where blur is less than blur_thres
for the given set of camera parameters
p=pixel size in m
N=f number (F-stop)
f=focal length in m
imgratio=(output image width in pixels)/(input image width in pixels)  . This is assumed to be 1 in blender dataset
s2range=the range of interested s2values [s2min,s2max]
s1range=the range of interested s1values
blur_thres=maximum values of blur allowed
blur is calculated as
blur=abs(s2-s1)/s2*1/(s1-f)*1/kcam*
1/kcam=f^2/N*1/p*imgratio
'''
p=3.1e-3/256
N=1
f=2.9e-3
s1range=[0.1,1.5]
s2range=[0.1,1.9]
imgratio=1
blur_thres=2.


blurs=[]
ind=[]
s1=0.15

for s2 in np.arange(s2range[0],s2range[1]+0.05,0.05):
    b=abs(s2-s1)/s2*1/(s1-f)*f**2/N*1/p*imgratio
    blurs.append(b)
    ind.append((s1,s2))
s2s=[i[1] for i in ind]
#plt.scatter(s2s,blurs)
#plt.show()  

for s2 in np.arange(s2range[0],s2range[1]+0.05,0.05):
    for s1 in np.arange(s1range[0],s1range[1]+0.05,0.05):
        b=abs(s2-s1)/s2*1/(s1-f)*f**2/N*1/p*imgratio
        blurs.append(b)
        ind.append((s1,s2))


def get_workable_s1s2ranges(p,N,f,s2range,s1range,blur_thres,imgratio=1):
    blur=[]
    ind=[]
    for s2 in np.arange(s2range[0],s2range[1]+0.05,0.05):
        for s1 in np.arange(s1range[0],s1range[1]+0.05,0.05):
            kcam=1/(f**2/N/p)
            b=abs(s2-s1)/s2/(s1-f)/kcam
            blur.append(b)
            ind.append((s1,s2))

    #check which values are under the threshold
    blur_np=np.array(blur)
    good_ind=np.argwhere(blur_np<blur_thres)[:,0]
    ind=np.array(ind)
    ind=np.around(ind,decimals=2)
    good_vals=ind[good_ind,:]
    unique_s1=np.unique(good_vals[:,0])
    print('Workable s1 and s2 ranges for the given camera: ')
    for s1 in unique_s1:
        vals=good_vals[good_vals[:,0]==s1]
        if(len(vals)>1):
            print('s1='+str(s1)+' min s2='+str(np.min(vals[:,1]))+' max s2='+str(np.max(vals[:,1])))


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







