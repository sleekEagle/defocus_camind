#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:54:12 2018
@author: maximov
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import transforms, utils

import numpy as np
import importlib
import random
import math
from sacred import Experiment
import csv
import util_func
import argparse
from dataloaders import DDFF12,focalblender

TRAIN_PARAMS = {
    'ARCH_NUM': 3,
    'FILTER_NUM': 16,
    'LEARNING_RATE': 0.0001,
    'FLAG_GPU': True,
    'EPOCHS_NUM': 100, 'EPOCH_START': 0,
    'RANDOM_LEN_INPUT': 0,
    'TRAINING_MODE':2, #1: do not use step 1 , 2: use step 2

    'MODEL_STEPS': 1,

    'MODEL1_LOAD': False,
    'MODEL1_ARCH_NUM': 3,
    'MODEL1_NAME': "dmediumN1_t01", 'MODEL1_INPUT_NUM': 1,
    'MODEL1_EPOCH': 0, 'MODEL1_FILTER_NUM': 16,
    'MODEL1_LOSS_WEIGHT': 1,

    'MODEL2_LOAD': False,
    'MODEL2_NAME': "a01_d06_t01",
    'MODEL2_EPOCH': 500,
    'MODEL2_TRAIN_STEP': True,
}

parser = argparse.ArgumentParser(description='camIndDefocus')
parser.add_argument('--blenderpth', default='C:\\usr\\wiss\\maximov\\RD\\DepthFocus\\Datasets\\focal_data\\', help='blender data path')
parser.add_argument('--bs', type=int,default=20, help='training batch size')
parser.add_argument('--depthscale', default=1.9,help='divide all depths by this value')
parser.add_argument('--fscale', default=1.9,help='divide all focal distances by this value')
parser.add_argument('--blurclip', default=90.0,help='Clip blur by this value : only applicable for camind model. Default=10')
parser.add_argument('--blurweight', default=0.3,help='weight for blur loss')
#parser.add_argument('--savedmodel', default='C:\\Users\\lahir\\code\\defocus\\models\\a03_exp01\\a03_exp01_ep0.pth', help='path to the saved model')
parser.add_argument('--savedmodel', default=None, help='path to the saved model')
parser.add_argument('--s2limits', nargs='+', default=[0.1,3.],  help='the interval of depth where the errors are calculated')
parser.add_argument('--dataset', default='blender', help='blender data path')
parser.add_argument('--camind', type=bool,default=True, help='True: use camera independent model. False: use defpcusnet model')
parser.add_argument('--aif', type=bool,default=False, help='True: Train with the AiF images. False: Train with blurred images')
args = parser.parse_args()

if(args.aif):
    expname='aif_N1_d_'+str(args.depthscale)
    TRAIN_PARAMS['ARCH_NUM']=4
else:
    if(args.camind):
        expname='camind_N1.9_f25mm_d_'+str(args.depthscale)+'_f'+str(args.fscale)+'_blurclip'+str(args.blurclip)+'_blurweight'+str(args.blurweight)
    else:
        expname='defocus_N1_d_'+str(args.depthscale)+'_f'+str(args.fscale)+'_blurweight'+str(args.blurweight)

    

OUTPUT_PARAMS = {
    'RESULT_PATH': 'C:\\Users\\lahir\\code\\defocus\\results\\',
    'MODEL_PATH': 'C:\\Users\\lahir\\code\\defocus\\models\\',
    'EXP_NAME':expname,
}

# ============ init ===============
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)

def train_model(loaders, model_info):
    criterion = torch.nn.MSELoss()
    #criterion=F.smooth_l1_loss(reduction='none')
    optimizer = optim.Adam(model_info['model_params'], lr=TRAIN_PARAMS['LEARNING_RATE'])

    ##### Training
    print("Total number of epochs:", TRAIN_PARAMS['EPOCHS_NUM'])
    for e_iter in range(TRAIN_PARAMS['EPOCHS_NUM'] - TRAIN_PARAMS['EPOCH_START']):
        epoch_iter = e_iter + TRAIN_PARAMS['EPOCH_START']
        loss_sum, iter_count,absloss_sum= 0,0,0
        depthloss_sum,blurloss_sum=0,0
        blur_sum=0
        mean_blur=0
        for st_iter, sample_batch in enumerate(loaders[0]):
            # Setting up input and output data
            X = sample_batch['input'][:,0,:,:,:].float().to(model_info['device_comp'])
            Y = sample_batch['output'].float().to(model_info['device_comp'])
            optimizer.zero_grad()
            
            #blur (|s2-s1|/(s2*(s1-f)))
            gt_step1 = Y[:, :-1, :, :]
            #depth in m
            gt_step2 = Y[:, -1:, :, :]
            
            mask=(gt_step2>args.s2limits[0]).int()*(gt_step2<args.s2limits[1]).int()

            mean_blur_=torch.sum

            # we only use focal stacks with a single image
            stacknum = 1

            '''
            convert blur_pix that is being estimated by the model
            into
            |s2-s1|/s1  by multiplying blur_pix_est by (s1-f)/kcam
            which is camera independent because it only depends on the distances; not on camera parameters.
            Also see dofNet_arch3.py comments on the calculations
            '''
            X2_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
            s1_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
            for t in range(stacknum):
                #iterate through the batch
                for i in range(X.shape[0]):
                    focus_distance=sample_batch['fdist'][i].item()
                    f=sample_batch['f'][i].item()
                    #X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :] * (focus_distance-sample_batch['f'][i].item())*sample_batch['kcam'][i].item()
                    #X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*(focus_distance-sample_batch['f'][i].item())/args.fscale
                    if(not args.aif):
                        X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*sample_batch['kcam'][i].item()*(focus_distance-f)/args.fscale
                        s1_fcs[i, t:(t + 1), :, :] = s1_fcs[i, t:(t + 1), :, :]*(focus_distance)/args.fscale

            X2_fcs = X2_fcs.float().to(model_info['device_comp'])
            s1_fcs = s1_fcs.float().to(model_info['device_comp'])
            #print('fdist:'+str(sample_batch['fdist']))
            # Forward and compute loss
            if(args.aif):
                output_step1,output_step2= util_func.forward_pass(X, model_info,stacknum=stacknum,camind=args.camind,camparam=X2_fcs,foc_dist=s1_fcs,aif=args.aif)
            else:
                output_step1,output_step2,_ = util_func.forward_pass(X, model_info,stacknum=stacknum,camind=args.camind,camparam=X2_fcs,foc_dist=s1_fcs,aif=args.aif)
            #print('mean blur pred:'+str(torch.mean(output_step1))+' min:'+str(torch.min(output_step1))+' max:'+str(torch.max(output_step1)))
            #print('mean gt blur:'+str(torch.mean(gt_step1))+' min:'+str(torch.min(gt_step1))+' max:'+str(torch.max(gt_step1)))
            #print('mean gt depth:'+str(torch.mean(gt_step2))+' min:'+str(torch.min(gt_step2))+' max:'+str(torch.max(gt_step2)))
            #output_step1=output_step1*(0.1-2.9e-3)*7
            blur_sum+=torch.sum(output_step1*mask).item()/torch.sum(mask)
            #blurpred=output_step1*(0.1-2.9e-3)*1.4398*7
            depth_loss=criterion(output_step2*mask, gt_step2/args.depthscale*mask)
            #we don't train blur if input images are AiF
            if(args.aif):
                blur_loss=0 
            else:
                blur_loss=criterion(output_step1*mask, gt_step1*mask)
            loss=depth_loss+blur_loss*args.blurweight

            absloss=torch.sum(torch.abs(output_step1-gt_step1)*mask)/torch.sum(mask)
            absloss_sum+=absloss.item()

            loss.backward()
            optimizer.step()

            # Training log
            loss_sum += loss.item()
            iter_count += 1.
            if(args.aif):
                blurloss_sum+=0
            else:
                blurloss_sum+=blur_loss.item()
            depthloss_sum+=depth_loss.item()

            if (st_iter + 1) % 10 == 0:
                print(model_info['model_name'], 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch_iter + 1, TRAIN_PARAMS['EPOCHS_NUM'], st_iter + 1, model_info['total_steps'], loss_sum / iter_count))
    
                absloss_sum=0
                depth_sum,blur_sum=0,0
                depthloss_sum,blurloss_sum=0,0

                total_iter = model_info['total_steps'] * epoch_iter + st_iter
                loss_sum, iter_count = 0,0

        # Save model
        if (epoch_iter+1) % 10 == 0:
            print('saving model')
            torch.save(model_info['model'].state_dict(), model_info['model_dir'] + model_info['model_name'] + '_ep' + str(0) + '.pth')
            s2loss1,s2loss2,blurloss,meanblur,gtmeanblur,minblur,maxblur=util_func.eval(loaders[1],model_info,dataset=args.dataset,camind=args.camind,
            depthscale=args.depthscale,fscale=args.fscale,s2limits=args.s2limits,aif=args.aif)
            print('s2 loss2: '+str(s2loss2))
            print('blur loss = '+str(blurloss))
            print('mean blur = '+str(meanblur))

def main():
    # Initial preparations
    model_dir, model_name = util_func.set_output_folders(OUTPUT_PARAMS, TRAIN_PARAMS)
    device_comp = util_func.set_comp_device(TRAIN_PARAMS['FLAG_GPU'])

    # Training initializations
    loaders, total_steps = focalblender.load_data(args.blenderpth,blur=1,aif=args.aif,train_split=0.8,fstack=0,WORKERS_NUM=0,
    BATCH_SIZE=args.bs,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,100000],REQ_F_IDX=[0,1,2,3,4],MAX_DPT=1.0,
    camind=args.camind,def_f_number=1,def_f=2.9e-3,blurclip=args.blurclip)

    model, inp_ch_num, out_ch_num = util_func.load_model(TRAIN_PARAMS)
    model = model.to(device=device_comp)
    model_params = model.parameters()

    # loading weights of the first step
    if args.savedmodel:
        print('loading model....')
        print('model path :'+args.savedmodel)
        pretrained_dict = torch.load(args.savedmodel)
        model_dict = model.state_dict()
        for param_tensor in model_dict:
            for param_pre in pretrained_dict:
                if param_tensor == param_pre:
                    model_dict.update({param_tensor: pretrained_dict[param_pre]})
        model.load_state_dict(model_dict)

    if TRAIN_PARAMS['MODEL2_TRAIN_STEP'] == 2:
        model_params += list(model.parameters())

    model_info = {'model': model,
                  'model_dir': model_dir,
                  'model_name': model_name,
                  'total_steps': total_steps,
                  'inp_ch_num': inp_ch_num,
                  'out_ch_num':out_ch_num,
                  'device_comp': device_comp,
                  'model_params': model_params,
                  }
    print("inp_ch_num",inp_ch_num,"   out_ch_num",out_ch_num)

    # Run training
    train_model(loaders=loaders, model_info=model_info)

if __name__ == "__main__":
    main()

#datapath='C:\\Users\\lahir\\focalstacks\\datasets\\mediumN1\\'
#focalblender.get_data_stats(datapath,50)
'''
fdist of DDFF 
tensor([[0.2800, 0.2511, 0.2222, 0.1933, 0.1644, 0.1356, 0.1067, 0.0778, 0.0489,
         0.0200]])
'''


#plotting distribution of blur
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

f=3e-3
p=3.1e-3/256
s=1
N=1
for f in [3e-3,4e-3,5e-3]:
    s1 = np.random.uniform(0.1,1.5,1000)
    s2 = np.random.uniform(0.0,2.0,1000)
    s2=np.random.normal(loc=1.0,scale=0.1,size=1000)
    blur=np.abs(s1-s2)/s2*1/(s1-f) * f**2/N *1/p * s
    density = stats.gaussian_kde(blur)
    bins = np.linspace(0.1, 2.0, 1000)
    n,bins = np.histogram(np.array(blur), bins)
    plt.plot(bins, density(bins),label='f=%1.0fmm'%(f*1000))

ax = plt.gca()
# Hide X and Y axes label marks
ax.yaxis.set_tick_params(labelleft=False)
# Hide Y axes tick marks
ax.set_yticks([])
plt.legend()
plt.xlabel('Blur in pixles')
plt.ylabel('Density')
plt.savefig('blur_distF.png', dpi=500)
plt.show()





