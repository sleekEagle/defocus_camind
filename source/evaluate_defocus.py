import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import transforms, utils

import numpy as np
import importlib
import util_func_defocusnet
import util_func
import argparse
from dataloaders import DDFF12
import torch
import sys


TRAIN_PARAMS = {
    'ARCH_NUM': 1,
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
    #'DATA_PATH': 'C:\\Users\\***\\focalstacks\\datasets\\',
    #'DATA_SET': '',  
    #'DATA_NUM': 'mediumN1',
    'DATA_PATH': 'C:\\Users\\***\\focalstacks\\datasets\\',
    'DATA_SET': '',
    'DATA_NUM': 'mediumN1-3',
    'FLAG_NOISE': False,
    'FLAG_SHUFFLE': False,
    'INP_IMG_NUM': 1,
    'REQ_F_IDX':[0,1,2,3,4], # the index of the focal distance aquired from the dataset. -1 for random fdist.
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
    'RESULT_PATH': 'C:\\Users\\***\\code\\defocus\\results\\',
    'MODEL_PATH': 'C:\\Users\\***\\code\\defocus\\models\\a01_dmediumN1_t01\\a01_dmediumN1_t01_ep0.pth',
}

parser = argparse.ArgumentParser(description='camIndDefocus')
parser.add_argument('--dataset', default='ddff', help='blender data path')
parser.add_argument('--ddffpth', default='C:\\Users\\***\\focalstacks\\datasets\\my_dff_trainVal.h5', help='blender data path')
parser.add_argument('--blenderpth', default='C:\\Users\\***\\focalstacks\\datasets\\mediumN1-3\\', help='blender data path')
parser.add_argument('--s2limits', nargs='+', default=[0,1.0],  help='the interval of depth where the errors are calculated')
parser.add_argument('--gtscale', default=1.0,help='gt depth values has been divided by this value')
parser.add_argument('--modelscale', default=3.0,help='multiply predicted depth by this value')
args = parser.parse_args()

def forward_pass(X, model_info, TRAIN_PARAMS, DATA_PARAMS, stacknum=1, additional_input=None):
    flag_step2 = True if TRAIN_PARAMS['TRAINING_MODE']==2 else False
    outputs = model_info['model'](X, model_info['inp_ch_num'], stacknum, flag_step2=flag_step2, x2 = additional_input)
    return (outputs[1], outputs[0]) if TRAIN_PARAMS['TRAINING_MODE']==2 else (outputs, outputs)

def kcamwise_blur():
    device_comp = util_func_defocusnet.set_comp_device(TRAIN_PARAMS['FLAG_GPU'])
    loaders, total_steps = util_func_defocusnet.load_data(DATA_PARAMS['DATA_PATH'],DATA_PARAMS['DATA_SET'],DATA_PARAMS['DATA_NUM'],
        DATA_PARAMS['FLAG_SHUFFLE'],DATA_PARAMS['FLAG_IO_DATA'],DATA_PARAMS['TRAIN_SPLIT'],
        DATA_PARAMS['WORKERS_NUM'],DATA_PARAMS['BATCH_SIZE'],DATA_PARAMS['DATASET_SHUFFLE'],DATA_PARAMS['DATA_RATIO_STRATEGY'],
        DATA_PARAMS['FOCUS_DIST'],DATA_PARAMS['REQ_F_IDX'],
        DATA_PARAMS['F_NUMBER'],DATA_PARAMS['MAX_DPT'])
   
    model, inp_ch_num, out_ch_num = util_func_defocusnet.load_model("", "",TRAIN_PARAMS, DATA_PARAMS)
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
    mse_ar,s2mse=[],[]
    kcams_all,meanblur_all,mse_all=torch.empty(0),torch.empty(0),torch.empty(0)
    for st_iter, sample_batch in enumerate(loaders[0]):
        X = sample_batch['input'].float().to(model_info['device_comp'])
        Y = sample_batch['output'].float().to(model_info['device_comp'])
        if TRAIN_PARAMS['TRAINING_MODE'] == 2:
            gt_step1 = Y[:, :-1, :, :]
            gt_step2 = Y[:, -1:, :, :]
        if(True):
                mask=(gt_step2*args.gtscale>args.s2limits[0]).int()*(gt_step2*args.gtscale<args.s2limits[1]).int()
        else:
            mask=torch.ones_like(gt_step2)

        stacknum = DATA_PARAMS['INP_IMG_NUM']
        focus_dists = DATA_PARAMS['FOCUS_DIST']
        kcams=sample_batch['kcam']
        kcams_all=torch.cat((kcams_all,kcams))

        X2_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
        for t in range(stacknum):
            for i in range(X.shape[0]):
                focus_distance=sample_batch['fdist'][i].item()/1.5
                X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :] * (focus_distance)
        X2_fcs = X2_fcs.float().to(model_info['device_comp'])
        output_step1, output_step2 = forward_pass(X, model_info, TRAIN_PARAMS, DATA_PARAMS,stacknum=stacknum, additional_input=X2_fcs)

        meanblur=torch.mean(output_step1,dim=2).mean(dim=2)[:,0].detach().cpu()
        meanblur_all=torch.cat((meanblur_all,meanblur))
        mse=torch.sum(torch.square((output_step2*args.modelscale-gt_step2*args.gtscale)*mask),dim=2).sum(dim=2)[:,0].detach().cpu()/torch.sum(mask,dim=2).sum(dim=2)[:,0].detach().cpu()
        mse_all=torch.cat((mse_all,mse))

    labels=torch.zeros_like(kcams_all,dtype=torch.int64)

    #get kcam wise mean blur
    unique_kcams, _ = kcams_all.unique(dim=0, return_counts=True)
    for i in range(unique_kcams.shape[0]):
        indices=((kcams_all == unique_kcams[i].item()).nonzero(as_tuple=True)[0])
        labels[indices]=i
    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, meanblur_all)
    res = res / labels_count

    res_mse = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, mse_all)
    res_mse = res_mse / labels_count
    print(unique_kcams)
    print(res)
    print(res_mse)


def main():
    device_comp = util_func_defocusnet.set_comp_device(TRAIN_PARAMS['FLAG_GPU'])

    if(args.dataset=='blender'):
        loaders, total_steps = util_func_defocusnet.load_data(DATA_PARAMS['DATA_PATH'],DATA_PARAMS['DATA_SET'],DATA_PARAMS['DATA_NUM'],
            DATA_PARAMS['FLAG_SHUFFLE'],DATA_PARAMS['FLAG_IO_DATA'],DATA_PARAMS['TRAIN_SPLIT'],
            DATA_PARAMS['WORKERS_NUM'],DATA_PARAMS['BATCH_SIZE'],DATA_PARAMS['DATASET_SHUFFLE'],DATA_PARAMS['DATA_RATIO_STRATEGY'],
            DATA_PARAMS['FOCUS_DIST'],DATA_PARAMS['REQ_F_IDX'],
            DATA_PARAMS['F_NUMBER'],DATA_PARAMS['MAX_DPT'])
    if(args.dataset=='ddff'):
        DDFF12_train = DDFF12.DDFF12Loader(args.ddffpth, stack_key="stack_train", disp_key="disp_train", n_stack=10,
                                    min_disp=0.02, max_disp=0.28,fstack=0,idx_req=[1])
        DDFF12_val = DDFF12.DDFF12Loader(args.ddffpth, stack_key="stack_val", disp_key="disp_val", n_stack=10,
                                            min_disp=0.02, max_disp=0.28, b_test=False,fstack=0,idx_req=[6,5,4,3,2,1,0])
        DDFF12_train, DDFF12_val = [DDFF12_train], [DDFF12_val]

        dataset_train = torch.utils.data.ConcatDataset(DDFF12_train)
        dataset_val = torch.utils.data.ConcatDataset(DDFF12_val) # we use the model perform better on  DDFF12_val

        TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=0, batch_size=1, shuffle=True, drop_last=True)
        ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val, num_workers=0, batch_size=1, shuffle=False, drop_last=True)
        loaders=[TrainImgLoader,ValImgLoader]


    model, inp_ch_num, out_ch_num = util_func_defocusnet.load_model("", "",TRAIN_PARAMS, DATA_PARAMS)
    model = model.to(device=device_comp)
    model_params = model.parameters()

    # loading weights of the trained defocusnet model
    trained_model=OUTPUT_PARAMS['MODEL_PATH']
    pretrained_dict = torch.load(trained_model)
    model_dict = model.state_dict()
    for param_tensor in model_dict:
        for param_pre in pretrained_dict:
            if param_tensor == param_pre:
                model_dict.update({param_tensor: pretrained_dict[param_pre]})
    model.load_state_dict(model_dict)

    model_info = {'model': model,
                    'inp_ch_num': inp_ch_num,
                    'out_ch_num':out_ch_num,
                    'device_comp': device_comp,
                    'model_params': model_params,
                    }
    mse_ar,s2mse=[],[]
    for st_iter, sample_batch in enumerate(loaders[0]):
        sys.stdout.write("\r%d is done"%st_iter)
        sys.stdout.flush()
        if(args.dataset=='blender'):
            X = sample_batch['input'].float().to(model_info['device_comp'])
            Y = sample_batch['output'].float().to(model_info['device_comp'])
            if TRAIN_PARAMS['TRAINING_MODE'] == 2:
                gt_step1 = Y[:, :-1, :, :]
                gt_step2 = Y[:, -1:, :, :]
        if(args.dataset=='ddff'):
            img_stack, gt_disp, foc_dist=sample_batch
            X=img_stack.float().to(model_info['device_comp'])
            Y=gt_disp.float().to(model_info['device_comp'])
            gt_step2=Y

        mask=(gt_step2*args.gtscale>args.s2limits[0]).int()*(gt_step2*args.gtscale<args.s2limits[1]).int()

        stacknum = DATA_PARAMS['INP_IMG_NUM']
        X2_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
        for t in range(stacknum):
            for i in range(X.shape[0]):
                if(args.dataset=='blender'):
                    focus_distance=sample_batch['fdist'][i].item()
                if(args.dataset=='ddff'):
                    focus_distance=foc_dist[i].item()
                X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :] * (focus_distance)/1.5
        X2_fcs = X2_fcs.float().to(model_info['device_comp'])
        output_step1, output_step2 = forward_pass(X, model_info, TRAIN_PARAMS, DATA_PARAMS,stacknum=stacknum, additional_input=X2_fcs)
        mse2=torch.sum(torch.square(output_step2*args.modelscale-gt_step2*args.gtscale)*mask).item()/torch.sum(mask).item()
        s2mse.append(mse2)
    print('mse='+str(sum(s2mse)/len(s2mse)))
    if(args.dataset=='blender'):    
        kcamwise_blur()        

if __name__ == "__main__":
    main()

