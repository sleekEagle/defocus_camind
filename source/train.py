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

TRAIN_PARAMS = {
    'ARCH_NUM': 3,
    'FILTER_NUM': 16,
    'LEARNING_RATE': 0.0001,
    'FLAG_GPU': True,
    'EPOCHS_NUM': 500, 'EPOCH_START': 0,
    'RANDOM_LEN_INPUT': 0,
    'TRAINING_MODE':2, #1: do not use step 1 , 2: use step 2

    'MODEL_STEPS': 1,

    'MODEL1_LOAD': False,
    'MODEL1_ARCH_NUM': 1,
    'MODEL1_NAME': "d06_t01", 'MODEL1_INPUT_NUM': 1,
    'MODEL1_EPOCH': 0, 'MODEL1_FILTER_NUM': 16,
    'MODEL1_LOSS_WEIGHT': 1,

    'MODEL2_LOAD': False,
    'MODEL2_NAME': "a01_d06_t01",
    'MODEL2_EPOCH': 500,
    'MODEL2_TRAIN_STEP': True,
}
DATA_PARAMS = {
    'DATA_PATH': 'C:\\usr\\wiss\\maximov\\RD\\DepthFocus\\Datasets\\',
    'DATA_SET': 'fs_',
    'DATA_NUM': 'train_N1',
    'FLAG_NOISE': False,
    'FLAG_SHUFFLE': False,
    'INP_IMG_NUM': 1,
    'REQ_F_IDX': [0], # list of indices of the focal distance aquired from the dataset. [] for random fdist.
    'CRITICAL_S2': 0.1,
    'FLAG_IO_DATA': {
        'INP_RGB': True,
        'INP_COC': False,
        'INP_AIF': False,
        'INP_DIST':False,
        'OUT_COC': True, # model outputs the blur
        'OUT_DEPTH': True, # model outputs the depth
    },
    'TRAIN_SPLIT': 0.9,
    'DATASET_SHUFFLE': True,
    'WORKERS_NUM': 4,
    'BATCH_SIZE': 16,
    'DATA_RATIO_STRATEGY': 0,
    'FOCUS_DIST': [0.1,.15,.3,0.7,1.5,1000000],
    'F_NUMBER': 1.,
    'MAX_DPT': 3.,
}

OUTPUT_PARAMS = {
    'RESULT_PATH': 'C:\\Users\\lahir\\code\\defocus\\results\\',
    'MODEL_PATH': 'C:\\Users\\lahir\\code\\defocus\\models\\',
    'VIZ_PORT': 8098, 'VIZ_HOSTNAME': "http://localhost", 'VIZ_ENV_NAME':'main',
    'VIZ_SHOW_INPUT': True, 'VIZ_SHOW_MID': True,
    'EXP_NUM': 1,
    'COMMENT': "Default",
}

def forward_pass(X, model_info, TRAIN_PARAMS, DATA_PARAMS, stacknum=1, additional_input=None):
    flag_step2 = True if TRAIN_PARAMS['TRAINING_MODE']==2 else False
    outputs = model_info['model'](X, model_info['inp_ch_num'], stacknum, flag_step2=flag_step2, x2 = additional_input,parallel=False)
    return (outputs[1], outputs[0]) if TRAIN_PARAMS['TRAINING_MODE']==2 else (outputs, outputs)

def eval(loaders,model_info):
    mse_ar=[]
    s2mse=[]
    for st_iter, sample_batch in enumerate(loaders[1]):
        X = sample_batch['input'].float().to(model_info['device_comp'])
        Y = sample_batch['output'].float().to(model_info['device_comp'])
        if TRAIN_PARAMS['TRAINING_MODE'] == 2:
            gt_step1 = Y[:, :-1, :, :]
            gt_step2 = Y[:, -1:, :, :]
        stacknum = DATA_PARAMS['INP_IMG_NUM']
        focus_dists = DATA_PARAMS['FOCUS_DIST']

        if(True):
            mask=(gt_step2>0.15).int()*(gt_step2<1.0).int()
        else:
            mask=torch.ones_like(gt_step2)

        X2_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]],requires_grad=False)
        for t in range(stacknum):
            #iterate through the batch
            for i in range(X.shape[0]):
                focus_distance=sample_batch['fdist'][i].item()
                X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :] * (focus_distance-sample_batch['f'][i].item())*sample_batch['kcam'][i].item()/1.5
        X2_fcs = X2_fcs.float().to(model_info['device_comp'])
        
        output_step1,output_step2 = forward_pass(X, model_info, TRAIN_PARAMS, DATA_PARAMS,stacknum=stacknum, additional_input=X2_fcs)
        #output_step1=output_step1*(0.1-2.9e-3)*7
        blurpred=output_step1
        #calculate s2 provided that s2>s1
        s2est=0.1*1./(1-blurpred)
        #calculate MSE value
        mse=torch.sum(torch.square(s2est-gt_step2)*mask).item()/torch.sum(mask).item()
        #mse_val, ssim_val, psnr_val=util_func.compute_all_metrics(output_step2*mask,gt_step2*mask)
        mse_ar.append(mse)
        mse2=torch.sum(torch.square(output_step2-gt_step2)*mask).item()/torch.sum(mask).item()
        s2mse.append(mse2)

    return sum(mse_ar)/len(mse_ar),sum(s2mse)/len(s2mse)

def train_model(loaders, model_info):
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model_info['model_params'], lr=TRAIN_PARAMS['LEARNING_RATE'])

    focus_dists = DATA_PARAMS['FOCUS_DIST']

    ##### Training
    print("Total number of epochs:", TRAIN_PARAMS['EPOCHS_NUM'])
    for e_iter in range(TRAIN_PARAMS['EPOCHS_NUM'] - TRAIN_PARAMS['EPOCH_START']):
        epoch_iter = e_iter + TRAIN_PARAMS['EPOCH_START']
        loss_sum, iter_count,absloss_sum= 0,0,0
        depthloss_sum,blurloss_sum=0,0
        blur_sum=0
        for st_iter, sample_batch in enumerate(loaders[0]):
            # Setting up input and output data
            X = sample_batch['input'].float().to(model_info['device_comp'])
            Y = sample_batch['output'].float().to(model_info['device_comp'])
            optimizer.zero_grad()

            #calculate |s2-s1|/s1 to see if gt blur is correct
            '''
            s2=Y[:,1,:,:]
            s1=torch.ones_like(s2)*0.1
            blur=torch.abs(s1-s2)/s2

            torch.mean(torch.abs(blur- Y[:,0,:,:]))
            torch.mean(Y[:,0,:,:])
            torch.mean(blur)
            '''

            '''
            import matplotlib.pyplot as plt
            i=1
            #fdist
            sample_batch['fdist'][i]

            plt.imshow(X[i,0,:,:].cpu())
            plt.show()

            #blur
            plt.imshow(Y[i,0,:,:].cpu())
            plt.show()

            #depth
            plt.imshow(Y[i,1,:,:].cpu())
            plt.show()

            s2=Y[i,1,40,111].item()
            s1=sample_batch['fdist'][i].item()
            blur=abs(s1-s2)/s2

            Y[i,0,40,111].item()

            torch.mean(Y[:,0,:,:])
            '''
            if TRAIN_PARAMS['TRAINING_MODE'] == 2:
                #blur |s2-s1|/s2
                gt_step1 = Y[:, :-1, :, :]
                #depth in m
                gt_step2 = Y[:, -1:, :, :]
            
            if(True):
                mask=(gt_step2>0.1).int()*(gt_step2<3.0).int()
            else:
                mask=torch.ones_like(gt_step2)

            stacknum = DATA_PARAMS['INP_IMG_NUM']

            '''
            convert blur_pix that is being estimated by the model
            into
            |s2-s1|/s1  by multiplying blur_pix_est by (s1-f)/kcam
            which is camera independent because it only depends on the distances; not on camera parameters.
            Also see dofNet_arch3.py comments on the calculations
            '''
            X2_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
            for t in range(stacknum):
                #iterate through the batch
                for i in range(X.shape[0]):
                    focus_distance=sample_batch['fdist'][i].item()
                    X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :] * (focus_distance-sample_batch['f'][i].item())*sample_batch['kcam'][i].item()
            X2_fcs = X2_fcs.float().to(model_info['device_comp'])
            
            # Forward and compute loss
            output_step1,output_step2 = forward_pass(X, model_info, TRAIN_PARAMS, DATA_PARAMS,stacknum=stacknum,additional_input=X2_fcs)
            #output_step1=output_step1*(0.1-2.9e-3)*7
            blur_sum+=torch.sum(output_step1*mask).item()/torch.sum(mask)
            #blurpred=output_step1*(0.1-2.9e-3)*1.4398*7
            depth_loss=criterion(output_step2*mask, gt_step2*mask)
            blur_loss=criterion(output_step1*mask, gt_step1*mask)
            loss=depth_loss+blur_loss

            absloss=torch.sum(torch.abs(output_step1-gt_step1)*mask)/torch.sum(mask)
            absloss_sum+=absloss.item()

            loss.backward()
            optimizer.step()

            # Training log
            loss_sum += loss.item()
            iter_count += 1.
            blurloss_sum+=blur_loss.item()
            depthloss_sum+=depth_loss.item()

            if (st_iter + 1) % 10 == 0:
                print(model_info['model_name'], 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch_iter + 1, TRAIN_PARAMS['EPOCHS_NUM'], st_iter + 1, model_info['total_steps'], loss_sum / iter_count))
                #print('abs loss: '+str(absloss_sum/iter_count))
                #print('mean blur: '+str(blur_sum/iter_count))
                s2loss1,s2loss2=eval(loaders,model_info)
                #print('s2 loss1: '+str(s2loss1))
                print('s2 loss2: '+str(s2loss2))
                print('depth loss = '+str(depthloss_sum/iter_count))
                print('blur loss = '+str(blurloss_sum/iter_count))
                absloss_sum=0
                depth_sum,blur_sum=0,0
                depthloss_sum,blurloss_sum=0,0

                total_iter = model_info['total_steps'] * epoch_iter + st_iter
                loss_sum, iter_count = 0,0

        # Save model
        if (epoch_iter + 1) % 10 == 0:
            print('saving model')
            torch.save(model_info['model'].state_dict(), model_info['model_dir'] + model_info['model_name'] + '_ep' + str(0) + '.pth')

def main():
    # Initial preparations
    model_dir, model_name, res_dir = util_func.set_output_folders(OUTPUT_PARAMS, DATA_PARAMS, TRAIN_PARAMS)
    device_comp = util_func.set_comp_device(TRAIN_PARAMS['FLAG_GPU'])

    # Training initializations
    loaders, total_steps = util_func.load_data(DATA_PARAMS['DATA_PATH'],DATA_PARAMS['DATA_SET'],DATA_PARAMS['DATA_NUM'],
    DATA_PARAMS['FLAG_SHUFFLE'],DATA_PARAMS['FLAG_IO_DATA'],DATA_PARAMS['TRAIN_SPLIT'],
    DATA_PARAMS['WORKERS_NUM'],DATA_PARAMS['BATCH_SIZE'],DATA_PARAMS['DATASET_SHUFFLE'],DATA_PARAMS['DATA_RATIO_STRATEGY'],
    DATA_PARAMS['FOCUS_DIST'],DATA_PARAMS['REQ_F_IDX'],
    DATA_PARAMS['F_NUMBER'],DATA_PARAMS['MAX_DPT'])

    model, inp_ch_num, out_ch_num = util_func.load_model(model_dir, model_name,TRAIN_PARAMS, DATA_PARAMS)
    model = model.to(device=device_comp)
    model_params = model.parameters()

    # loading weights of the first step
    if TRAIN_PARAMS['TRAINING_MODE']==2 and TRAIN_PARAMS['MODEL1_LOAD']:
        model_dir1 = OUTPUT_PARAMS['MODEL_PATH']
        model_name1 = 'a' + str(TRAIN_PARAMS['MODEL1_ARCH_NUM']).zfill(2) + '_' + TRAIN_PARAMS['MODEL1_NAME']
        print("model_name1", model_dir1, model_name1)
        pretrained_dict = torch.load( model_dir1 + model_name1+'/'+model_name1 + '_ep' + str(TRAIN_PARAMS['MODEL1_EPOCH']) + '.pth')
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

'''
import numpy as np
import matplotlib.pyplot as plt

s1ar=np.arange(0.1,1.5,0.047)
s2ar=np.arange(0.1,0.8,0.01)


blurs=[]
s1list=[]
s2list=[]
for s1 in s1ar:
    for s2 in s2ar:
        blurs.append(abs(s1-s2)/s2)
        s1list.append(s1)
        s2list.append(s2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(s1list, s2list, blurs, 'gray')
plt.xlabel('s1')
plt.ylabel('s2')
plt.zlabel('|s1-s2|/s2')
plt.show()




blur=abs(s1-s2)/s2
min(blur),max(blur)

plt.hist(blurs)
plt.show()
min(blurs),max(blurs)
'''



