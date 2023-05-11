#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:54:12 2018
@author: maximov
"""

import torch
import torch.nn as nn
import torch.utils.data

from os import mkdir
from os.path import isdir
import numpy as np
import importlib
import csv
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

NORM_MIN=0.066
NORM_MAX=28.0
def normalize(x):
    v=(x-NORM_MIN)/(NORM_MAX-NORM_MIN)
    return v
def denormalize(v):
    x=v*(NORM_MAX-NORM_MIN)+NORM_MIN
    return x

def eval(model,loader,args,device_comp,kcam_in=0,f_in=0,fd_in=0,calc_distmse=False):
    meanMSE,meanMSE2,meanblurmse,meanblur=0,0,0,0
    minblur,maxblur,gt_meanblur=100,0,0
    #store distance wise mse
    distmse,distsum,distblur=torch.zeros(100),torch.zeros(100),torch.zeros(100)

    c=0

    print('Total samples = '+str(len(loader)))
    for st_iter, sample_batch in enumerate(loader):
        #if(st_iter>100):
        #    break
        #sys.stdout.write(str(st_iter)+" of "+str(len(loader))+" is done")
        sys.stdout.write("\r%d is done"%st_iter)
        sys.stdout.flush()

        if(args.dataset=='ddff'):
            img_stack, gt_disp, foc_dist=sample_batch
            X=img_stack.float().to(device_comp)
            Y=gt_disp.float().to(device_comp)
            gt_step2=Y
        elif(args.dataset=='blender' or args.dataset=='defocusnet'):
            # Setting up input and output data
            X = sample_batch['input'][:,0,:,:,:].float().to(device_comp)
            depth=sample_batch['depth'].float().to(device_comp)
            blur=sample_batch['blur'].float().to(device_comp)
            focus_distance=sample_batch['fdist']
            focus_distance=torch.unsqueeze(focus_distance,dim=2).unsqueeze(dim=3)
            focus_distance=torch.repeat_interleave(focus_distance,depth.shape[2],dim=2).repeat_interleave(depth.shape[3],dim=3)
            focus_distance=focus_distance.to(device_comp)
        elif(args.dataset=="nyu" or args.dataset=="DSLR"):
            X=sample_batch['rgb'].float().to(device_comp)
            depth=sample_batch['depth'].float().to(device_comp)
            blur=sample_batch['blur'].float().to(device_comp)
            depth=torch.unsqueeze(depth,dim=1)
            depth=torch.unsqueeze(depth,dim=1)
            focus_distance=sample_batch['fdist']
            focus_distance=torch.unsqueeze(focus_distance,dim=2).unsqueeze(dim=3)
            focus_distance=torch.repeat_interleave(focus_distance,depth.shape[2],dim=2).repeat_interleave(depth.shape[3],dim=3)
            focus_distance=focus_distance.to(device_comp)

        if(len(args.s2limits)==2):
            if(args.out_depth==1):
                 mask=(depth>args.s2limits[0])*(depth<args.s2limits[1]).int()
            else:
                mask=((focus_distance*depth)>args.s2limits[0])*((focus_distance*depth)<args.s2limits[1]).int()
            s=torch.sum(mask).item()
            #continue loop if there are no ground truth data in the range we are interested in
            if(s==0):
                print('no data in the provided range for this batch. skipping.')
                continue
        else:
            mask=torch.ones_like(depth)
        
        stacknum = 1
        X2_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
        s1_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
        s1_fcs = s1_fcs.float().to(device_comp)
        for t in range(stacknum):
            #iterate through the batch
            for i in range(X.shape[0]):
                if(args.dataset=='blender'or args.dataset=='defocusnet' or args.dataset=='nyu'):
                    fd=sample_batch['fdist'][i].item()
                    f=sample_batch['f'][i].item()
                    k=sample_batch['kcam'][i].item()
                #use manually input camera parameters
                elif(args.dataset=="DSLR"):
                    fd=fd_in
                    f=f_in
                    k=kcam_in
                    # print('kcam:'+str(k))
                    # print('f:'+str(f))
                    # print('fd:'+str(fd))
                if(not args.aif):
                    X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*k*(fd-f)
                    s1_fcs[i, t:(t + 1), :, :] = s1_fcs[i, t:(t + 1), :, :]*(focus_distance)
                elif(args.dataset=='ddff'):
                    fd=foc_dist[i].item()
                    if(not args.aif):
                        X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*kcam*(fd-f)
                        s1_fcs[i, t:(t + 1), :, :] = s1_fcs[i, t:(t + 1), :, :]*(focus_distance)
        X2_fcs = X2_fcs.float().to(device_comp)
        if(args.out_depth==1):
                pred_depth,pred_blur,corrected_blur=model(X,camind=args.camind,camparam=X2_fcs,foc_dist=s1_fcs)
        else:
            pred_depth,pred_blur,corrected_blur=model(X,camind=args.camind,camparam=X2_fcs)

        #scale the predictions back
        pred_depth*=args.depthscale
        
        #gt_blur=torch.cat((gt_blur,torch.flatten(gt_step1.detach().cpu())))
        #pred_blur=torch.cat((pred_blur,torch.flatten(output_step1.detach().cpu())))

        minblur_=torch.min(pred_blur).item()
        if(minblur_<minblur):   
            minblur=minblur_
        maxblur_=torch.max(pred_depth).item()
        if(maxblur_>maxblur):
            maxblur=maxblur_

        #output_step1=output_step1*(0.1-2.9e-3)*7 
        blurpred=pred_blur
        #calculate s2 provided that s2>s1
        s2est=0.1*1./(1-blurpred)
        #blur mse
        if(args.dataset=='blender' or args.dataset=='defocusnet' or args.dataset=='nyu'):
            if(args.dataset=='nyu'):
                blur_=torch.unsqueeze(blur,dim=0)
                mask_=torch.squeeze(mask,dim=0)
            else:
                blur_=blur
                mask_=mask
            blurmse=torch.mean(torch.square(pred_blur*args.blurclip-blur_)[mask_>0]).item()
            meanblurmse+=blurmse
        #calculate MSE value
        if(args.out_depth==1):
            mse=torch.mean(torch.square(pred_depth*args.depthscale-depth)[mask>0]).item()
        else:
            mse=torch.mean(torch.square(focus_distance*pred_depth-focus_distance*depth)[mask>0]).item()
            mse2=torch.mean(torch.square(pred_depth-depth)[mask>0]).item()
            meanMSE2+=mse2
        meanMSE+=mse
        c+=1

        if(calc_distmse):
            if(args.dataset=='nyu'):
                depth=torch.squeeze(depth,dim=0)
                pred_depth=torch.squeeze(pred_depth,dim=0)
            squareder=torch.square(pred_depth*focus_distance-depth*focus_distance)
            gtround=torch.round(depth*focus_distance*10,decimals=0)
            
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

        blur=torch.sum(pred_blur*mask).item()/torch.sum(mask).item()
        meanblur+=blur
        if(args.dataset=='blender' or args.dataset=='defocusnet'):
            gtblur=torch.sum(blur*mask).item()/torch.sum(mask).item()
            gt_meanblur+=gtblur
    if(calc_distmse):
        print('\ndistance wise error (distances rounded to the shown value): ')
        mse_=distmse/distsum
        blur_=distblur/distsum
        # mse_=mse_[~torch.isnan(mse_)]
        # blur_=blur_[~torch.isnan(blur_)]
        values=np.arange(0.1,(len(mse_)+1)*0.1,0.1)
        values=values[:-1]
        print('distances:')
        for i,v in enumerate(values):
            print("%4.3f"%(v),end=",")
        print('\nMSE:')
        for i,v in enumerate(values):
            print("%4.3f"%(mse_[i].item()),end=",")
        print('\nRMSE:')
        for i,v in enumerate(values):
            print("%4.3f"%((mse_[i].item())**0.5),end=",")
        print('')
    print("c:"+str(c))
    print("loader:"+str(len(loader)))
    return meanMSE/c,meanMSE2/c,meanblurmse/c,meanblur/c,gt_meanblur/c,minblur,maxblur

def kcamwise_blur(model,loader,args,device_comp):
    print('iscamind:'+str(args.camind))
    meanblur,meanblur_corrected=0,0
    kcams_all,meanblur_all,meanblur_corrected_all,mse_all=torch.empty(0),torch.empty(0),torch.empty(0),torch.empty(0)
    for st_iter, sample_batch in enumerate(loader):
        #if(st_iter>100):
        #    break
        #sys.stdout.write(str(st_iter)+" of "+str(len(loader))+" is done")
        sys.stdout.write("\r%d is done"%st_iter)
        sys.stdout.flush()

        if(args.dataset=='ddff'):
            img_stack, gt_disp, foc_dist=sample_batch
            X=img_stack.float().to(device_comp)
            Y=gt_disp.float().to(device_comp)
            gt_step2=Y
        elif(args.dataset=='blender' or args.dataset=='defocusnet'):
            # Setting up input and output data
            X = sample_batch['input'][:,0,:,:,:].float().to(device_comp)
            depth=sample_batch['depth'].float().to(device_comp)
            blur=sample_batch['blur'].float().to(device_comp)
            focus_distance=sample_batch['fdist']
            focus_distance=torch.unsqueeze(focus_distance,dim=2).unsqueeze(dim=3)
            focus_distance=torch.repeat_interleave(focus_distance,depth.shape[2],dim=2).repeat_interleave(depth.shape[3],dim=3)
            focus_distance=focus_distance.to(device_comp)
        elif(args.dataset=="nyu" or args.dataset=="DSLR"):
            X=sample_batch['rgb'].float().to(device_comp)
            depth=sample_batch['depth'].float().to(device_comp)
            blur=sample_batch['blur'].float().to(device_comp)
            depth=torch.unsqueeze(depth,dim=1)
            depth=torch.unsqueeze(depth,dim=1)
            focus_distance=sample_batch['fdist']
            focus_distance=torch.unsqueeze(focus_distance,dim=2).unsqueeze(dim=3)
            focus_distance=torch.repeat_interleave(focus_distance,depth.shape[2],dim=2).repeat_interleave(depth.shape[3],dim=3)
            focus_distance=focus_distance.to(device_comp)

        if(len(args.s2limits)==2):
            if(args.out_depth==1):
                 mask=(depth>args.s2limits[0])*(depth<args.s2limits[1]).int()
            else:
                mask=((focus_distance*depth)>args.s2limits[0])*((focus_distance*depth)<args.s2limits[1]).int()
            s=torch.sum(mask).item()
            #continue loop if there are no ground truth data in the range we are interested in
            if(s==0):
                print('no data in the provided range for this batch. skipping.')
                continue
        else:
            mask=torch.ones_like(depth)
        
        stacknum = 1
        X2_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
        s1_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
        s1_fcs = s1_fcs.float().to(device_comp)
        for t in range(stacknum):
            #iterate through the batch
            for i in range(X.shape[0]):
                if(args.dataset=='blender'or args.dataset=='defocusnet' or args.dataset=='nyu'):
                    fd=sample_batch['fdist'][i].item()
                    f=sample_batch['f'][i].item()
                    k_=sample_batch['kcam'][i]
                    k_=torch.unsqueeze(k_,dim=0)
                    kcams_all=torch.cat((kcams_all,k_))
                    k=k_.item()
                if(not args.aif):
                    X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*k*(fd-f)
                    s1_fcs[i, t:(t + 1), :, :] = s1_fcs[i, t:(t + 1), :, :]*(focus_distance)
                elif(args.dataset=='ddff'):
                    fd=foc_dist[i].item()
                    if(not args.aif):
                        X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*kcam*(fd-f)
                        s1_fcs[i, t:(t + 1), :, :] = s1_fcs[i, t:(t + 1), :, :]*(focus_distance)
        X2_fcs = X2_fcs.float().to(device_comp)
        if(args.out_depth==1):
                pred_depth,pred_blur,corrected_blur=model(X,camind=args.camind,camparam=X2_fcs,foc_dist=s1_fcs)
        else:
            pred_depth,pred_blur,corrected_blur=model(X,camind=args.camind,camparam=X2_fcs)

        #scale the predictions back
        pred_depth*=args.depthscale

        meanblur=torch.unsqueeze(torch.mean(pred_blur[mask>0]).detach().cpu(),dim=0)
        meanblur_corrected=torch.unsqueeze(torch.mean(corrected_blur[mask>0]).detach().cpu(),dim=0)
        meanblur_corrected_all=torch.cat((meanblur_corrected_all,meanblur_corrected))

        meanblur_all=torch.cat((meanblur_all,meanblur))
        mse=torch.unsqueeze(torch.mean(torch.square((pred_depth*args.depthscale-depth)[mask>0])),dim=0).detach().cpu()
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
    
    print('kcams:')
    for i in unique_kcams.tolist():
        print('%2.3f,'%(i),end=" ")
    print('\nmean blurs:')
    for i in blurres.tolist():
        print('%2.3f,'%(i),end=" ")
    print('\nMSE:')
    for i in mseres.tolist():
        print('%2.3f,'%(i),end=" ")
    print('')

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
p=36e-6
N=1.0
s2range=[0.1,10]
s1range=[1,10]
f=50.0e-3
blur_thres=6.9
def get_workable_s1s2ranges(p,N,f,s2range,s1range,blur_thres,imgratio=1):
    s1list,s2list,blur=[],[],[]
    kcam=1/(f**2/N/p)
    for s2 in np.arange(s2range[0],s2range[1]+0.05,0.05):
        for s1 in np.arange(s1range[0],s1range[1]+0.05,0.05):
            b=abs(s2-s1)/s2/(s1-f)/kcam
            s1list.append(s1)
            s2list.append(s2)
            blur.append(b)
            '''
            if(b>blur_thres):
                col.append('red')
            else:
                col.append('green')
            '''
    s1list=np.array(s1list)
    s2list=np.array(s2list)
    blur=np.array(blur)
    #plot in range values
    plt.scatter(s1list[blur<blur_thres],s2list[blur<blur_thres],c='green',marker='.')
    #plot out of range values
    plt.scatter(s1list[blur>=blur_thres],s2list[blur>=blur_thres],c='red',marker='x')
    plt.xlabel('Object Distance S2 - m')
    plt.ylabel('Focal Distance S1 - m')
    plt.title('Workable reange of the camera kcam=%2.2f f=%2.1fmm'%(kcam,f*1000))
    plt.savefig('workablerange.png', dpi=500)
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







