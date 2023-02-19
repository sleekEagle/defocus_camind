import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import transforms, utils

import numpy as np
import importlib
import util_func
import argparse
from dataloaders import DDFF12,focalblender



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


OUTPUT_PARAMS = {
    'RESULT_PATH': 'C:\\Users\\lahir\\code\\defocus\\results\\',
    'MODEL_PATH': 'C:\\Users\\lahir\\code\\defocus\\models\\a03_exp01\\a03_exp01_ep0.pth',
    'EXP_NUM': 1,
}

parser = argparse.ArgumentParser(description='camIndDefocus')
parser.add_argument('--blenderpth', default='C:\\Users\\lahir\\focalstacks\\datasets\\mediumN1\\', help='blender data path')
parser.add_argument('--ddffpth', default='C:\\Users\\lahir\\focalstacks\\datasets\\my_dff_trainVal.h5', help='blender data path')
parser.add_argument('--dataset', default='blender', help='blender data path')
parser.add_argument('--bs', type=int,default=1, help='training batch size')
parser.add_argument('--depthscale', default=1.9,help='divide all depths by this value')
parser.add_argument('--fscale', default=1.9,help='divide all focal distances by this value')
parser.add_argument('--savedmodel', default='C:\\Users\\lahir\\code\\defocus\\models\\a03_exp01\\a03_exp01_ep0.pth', help='path to the saved model')
parser.add_argument('--kcamest', type=int,default=1,help='use the estimated kcam parameters')
parser.add_argument('--s2limits', nargs='+', default=[0,1.0],  help='the interval of depth where the errors are calculated')
args = parser.parse_args()

def main():
    device_comp = util_func.set_comp_device(TRAIN_PARAMS['FLAG_GPU'])
    #load the required dataset
    if(args.dataset=='blender'):
        loaders, total_steps = focalblender.load_data(args.blenderpth,blur=1,aif=0,train_split=1.,fstack=0,WORKERS_NUM=0,
        BATCH_SIZE=args.bs,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,100000],REQ_F_IDX=[0,1,2,3,4],MAX_DPT=1)
    elif(args.dataset=='ddff'):
        DDFF12_train = DDFF12.DDFF12Loader(args.ddffpth, stack_key="stack_train", disp_key="disp_train", n_stack=10,
                                    min_disp=0.02, max_disp=0.28,fstack=0,idx_req=[9])
        DDFF12_val = DDFF12.DDFF12Loader(args.ddffpth, stack_key="stack_val", disp_key="disp_val", n_stack=10,
                                            min_disp=0.02, max_disp=0.28, b_test=False,fstack=0,idx_req=[6,5,4,3,2,1,0])
        DDFF12_train, DDFF12_val = [DDFF12_train], [DDFF12_val]

        dataset_train = torch.utils.data.ConcatDataset(DDFF12_train)
        dataset_val = torch.utils.data.ConcatDataset(DDFF12_val) # we use the model perform better on  DDFF12_val

        TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=0, batch_size=1, shuffle=True, drop_last=True)
        ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val, num_workers=0, batch_size=1, shuffle=False, drop_last=True)


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

    model_info = {'model': model,
                    'inp_ch_num': inp_ch_num,
                    'out_ch_num':out_ch_num,
                    'device_comp': device_comp,
                    'model_params': model_params,
                    }
    if(args.dataset=='blender'):           
        s2loss1,s2loss2,blurloss,meanblur=util_func.eval(loaders[0],model_info,args.depthscale,args.fscale,args.s2limits,
        dataset=args.dataset)
        util_func.kcamwise_blur(loaders[0],model_info,args.depthscale,args.fscale,args.s2limits)
    elif(args.dataset=='ddff'):
        print('DDFF dataset Evaluation')
        s2loss1,s2loss2,blurloss,meanblur=util_func.eval(TrainImgLoader,model_info,args.depthscale,args.fscale,args.s2limits,
        dataset=args.dataset,kcam=5.2,f=3e-3)
    print('s2 loss2: '+str(s2loss2))
    print('blur loss = '+str(blurloss))
    print('mean blur = '+str(meanblur))        
     
if __name__ == "__main__":
    main()
    

