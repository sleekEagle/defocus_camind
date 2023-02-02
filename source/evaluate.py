import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import transforms, utils

import numpy as np
import importlib
import util_func


TRAIN_PARAMS = {
    'ARCH_NUM': 3,
    'FILTER_NUM': 16,
    'FLAG_GPU': True,
    'TRAINING_MODE': 2, #1: do not use step 1 , 2: use step 2
    'EPOCHS_NUM': 100, 'EPOCH_START': 0,

    'MODEL_STEPS': 1,

    'MODEL1_LOAD': False,
    'MODEL1_ARCH_NUM': 1,
    'MODEL1_NAME': "d06_t01", 'MODEL1_INPUT_NUM': 1,
    'MODEL1_EPOCH': 0, 'MODEL1_FILTER_NUM': 16,
    'MODEL1_LOSS_WEIGHT': 1.,
}

DATA_PARAMS = {
    'DATA_PATH': 'C:\\usr\\wiss\\maximov\\RD\\DepthFocus\\Datasets\\',
    'DATA_SET': 'fs_',
    'DATA_NUM': 'N1.35',
    'FLAG_NOISE': False,
    'FLAG_SHUFFLE': False,
    'INP_IMG_NUM': 1,
    'REQ_F_IDX':[0], # the index of the focal distance aquired from the dataset. -1 for random fdist.
    'CRITICAL_S2':0.1,
    'FLAG_IO_DATA': {
        'INP_RGB': True,
        'INP_COC': False,
        'INP_AIF': False,
        'INP_DIST':False,
        'OUT_COC': True, # model outputs the blur
        'OUT_DEPTH': True, # model outputs the depth
    },
    'TRAIN_SPLIT': 1.0,
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
    'MODEL_PATH': 'C:\\Users\\lahir\\code\\defocus\\models\\trainingrandN1.5\\ours_f0.1_dist0.15-3.0\\a03_dtrainingrandN1.5_t01_ep0.pth',
    'EXP_NUM': 1,
}

def forward_pass(X, model_info, TRAIN_PARAMS, DATA_PARAMS, stacknum=1, additional_input=None):
    flag_step2 = True if TRAIN_PARAMS['TRAINING_MODE']==2 else False
    outputs = model_info['model'](X, model_info['inp_ch_num'], stacknum, flag_step2=flag_step2, x2 = additional_input,parallel=False)
    return (outputs[1], outputs[0]) if TRAIN_PARAMS['TRAINING_MODE']==2 else (outputs, outputs)

def main(kcam):
    device_comp = util_func.set_comp_device(TRAIN_PARAMS['FLAG_GPU'])
    model_dir, model_name, res_dir = util_func.set_output_folders(OUTPUT_PARAMS, DATA_PARAMS, TRAIN_PARAMS)

    loaders, total_steps = util_func.load_data(DATA_PARAMS['DATA_PATH'],DATA_PARAMS['DATA_SET'],DATA_PARAMS['DATA_NUM'],
        DATA_PARAMS['FLAG_SHUFFLE'],DATA_PARAMS['FLAG_IO_DATA'],DATA_PARAMS['TRAIN_SPLIT'],
        DATA_PARAMS['WORKERS_NUM'],DATA_PARAMS['BATCH_SIZE'],DATA_PARAMS['DATASET_SHUFFLE'],DATA_PARAMS['DATA_RATIO_STRATEGY'],
        DATA_PARAMS['FOCUS_DIST'],DATA_PARAMS['REQ_F_IDX'],
        DATA_PARAMS['F_NUMBER'],DATA_PARAMS['MAX_DPT'])


    model, inp_ch_num, out_ch_num = util_func.load_model(model_dir, model_name,TRAIN_PARAMS, DATA_PARAMS)
    model = model.to(device=device_comp)
    model_params = model.parameters()

    # loading weights of the trained model
    trained_model=OUTPUT_PARAMS['MODEL_PATH']
    pretrained_dict = torch.load(trained_model)
    model_dict = model.state_dict()
    for param_tensor in model_dict:
        for param_pre in pretrained_dict:
            if param_tensor == param_pre:
                model_dict.update({param_tensor: pretrained_dict[param_pre]})
    model.load_state_dict(model_dict)

    model_info = {'model': model,
                    'total_steps': total_steps,
                    'inp_ch_num': inp_ch_num,
                    'out_ch_num':out_ch_num,
                    'device_comp': device_comp,
                    'model_params': model_params,
                    }
    absloss_sum,iter_count,blur_sum=0,0,0
    s1mse,s2mse,blurlosses,meanblurs=[],[],[],[]
    
    for st_iter, sample_batch in enumerate(loaders[0]):
        X = sample_batch['input'].float().to(model_info['device_comp'])
        Y = sample_batch['output'].float().to(model_info['device_comp'])
        if TRAIN_PARAMS['TRAINING_MODE'] == 2:
            gt_step1 = Y[:, :-1, :, :]
            gt_step2 = Y[:, -1:, :, :]
        stacknum = DATA_PARAMS['INP_IMG_NUM']
        focus_dists = DATA_PARAMS['FOCUS_DIST']
        if(True):
            mask=(gt_step2>0.15).int()*(gt_step2<0.5).int()
        else:
            mask=torch.ones_like(gt_step2)

        X2_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]],requires_grad=False)
        s1_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
        for t in range(stacknum):
            #iterate through the batch
            for i in range(X.shape[0]):
                focus_distance=sample_batch['fdist'][i].item()
                #print((focus_distance-sample_batch['f'][i].item())/sample_batch['kcam'][i].item())
                X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :] * (focus_distance-sample_batch['f'][i].item())*sample_batch['kcam'][i].item()/1.5
                s1_fcs[i, t:(t + 1), :, :] = s1_fcs[i, t:(t + 1), :, :] * (focus_distance)
                #X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]
        X2_fcs = X2_fcs.float().to(model_info['device_comp'])
        s1_fcs = s1_fcs.float().to(model_info['device_comp'])
        print(sample_batch['kcam'][0].item())
        print(sample_batch['fdist'][0].item())
        print('****')
        
        output_step1,output_step2 = forward_pass(X, model_info, TRAIN_PARAMS, DATA_PARAMS,stacknum=stacknum, additional_input=X2_fcs)
        blurpred=output_step1
        #calculate s2 provided that s2>s1
        s2est=0.1*1./(1-blurpred)
        #calculate MSE value
        mse1=torch.sum(torch.square(s2est-gt_step2)*mask).item()/torch.sum(mask).item()
        #mse_val, ssim_val, psnr_val=util_func.compute_all_metrics(output_step2*mask,gt_step2*mask)
        s1mse.append(mse1)
        mse2=torch.sum(torch.square(output_step2-gt_step2)*mask).item()/torch.sum(mask).item()
        s2mse.append(mse2)
        #mean blur
        blurloss=torch.sum(torch.square(output_step1-gt_step1)*mask).item()/torch.sum(mask).item()
        blurlosses.append(blurloss)
        meanblur=torch.sum(torch.square(output_step1)*mask).item()/torch.sum(mask).item()
        meanblurs.append(meanblur)
        iter_count+=1
    s2loss1=sum(s1mse)/len(s1mse)
    s2loss2=sum(s2mse)/len(s2mse)
    print('s2 loss1: '+str(s2loss1))
    print('s2 loss2: '+str(s2loss2))
    print('blur loss = '+str(sum(blurlosses)/len(blurlosses)))
    print('mean blur = '+str(sum(meanblurs)/len(meanblurs)))


if __name__ == "__main__":
    main(1)
    '''
    for i in range(1,10):
        print(i)
        main(i)
        print('\n\n')
    '''
    
    
    
    

