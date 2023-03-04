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
#parser.add_argument('--blenderpth', default="C://Users//lahir//focalstacks//datasets//mediumN1-10_test_remapped//", help='blender data path')
#parser.add_argument('--blenderpth', default="C://usr//wiss//maximov//RD//DepthFocus//Datasets//focal_data_remapped//", help='blender data path')
parser.add_argument('--blenderpth', default="C://Users//lahir//focalstacks//datasets//mediumN1//", help='blender data path')
parser.add_argument('--kcamfile', default=None, help='blender data path')
parser.add_argument('--ddffpth', default='C:\\Users\\lahir\\focalstacks\\datasets\\my_dff_trainVal.h5', help='blender data path')
parser.add_argument('--dataset', default='blender', help='blender data path')
parser.add_argument('--bs', type=int,default=1, help='training batch size')
parser.add_argument('--depthscale', default=1.9,help='divide all depths by this value')
parser.add_argument('--fscale', default=1.9,help='divide all focal distances by this value')
#parser.add_argument('--savedmodel', default='C:\\Users\\lahir\\code\\defocus\\models\\a03_expdefocus_d1.9_f1.9\\a03_expdefocus_d1.9_f1.9_ep0.pth', help='path to the saved model')
#parser.add_argument('--savedmodel', default='C:\\Users\\lahir\\code\\defocus\\models\\a04_expaif_N1_d_1.9\\a04_expaif_N1_d_1.9_ep0.pth', help='path to the saved model')
parser.add_argument('--savedmodel', default='C:\\Users\\lahir\\code\\defocus\\models\\a03_expcamind_fdistmul_N1_d_1.9_f1.9_blurclip8.0_blurweight0.3\\a03_expcamind_fdistmul_N1_d_1.9_f1.9_blurclip8.0_blurweight0.3_ep0.pth', help='path to the saved model')
parser.add_argument('--s2limits', nargs='+', default=[0.1,2.0],  help='the interval of depth where the errors are calculated')
parser.add_argument('--camind', type=bool,default=True, help='True: use camera independent model. False: use defocusnet model')
parser.add_argument('--aif', type=bool,default=False, help='True: Train with the AiF images. False: Train with blurred images')
args = parser.parse_args()

if(args.aif):
    TRAIN_PARAMS['ARCH_NUM']=4

def main():
    device_comp = util_func.set_comp_device(TRAIN_PARAMS['FLAG_GPU'])
    #load the required dataset
    if(args.dataset=='blender'):
        if(args.kcamfile):
            kcampath=args.blenderpth+args.kcamfile
        else:
            kcampath=None
        loaders, total_steps = focalblender.load_data(args.blenderpth,blur=1,aif=args.aif,train_split=0.8,fstack=0,WORKERS_NUM=0,
        BATCH_SIZE=args.bs,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,100000],REQ_F_IDX=[4],MAX_DPT=1,kcampath=kcampath)
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
        print('evaluating on blender')         
        s2loss1,s2loss2,blurloss,meanblur,gtmeanblur,minblur,maxblur=util_func.eval(loaders[1],model_info,args.depthscale,args.fscale,args.s2limits,
        dataset=args.dataset,camind=args.camind,aif=args.aif,calc_distmse=True)
        #util_func.kcamwise_blur(loaders[0],model_info,args.depthscale,args.fscale,args.s2limits,camind=args.camind,aif=args.aif)
    elif(args.dataset=='ddff'):
        print('DDFF dataset Evaluation')
        kcam=37
        s2loss1,s2loss2,blurloss,meanblur,gtmeanblur,minblur,maxblur=util_func.eval(TrainImgLoader,model_info,args.depthscale,args.fscale,args.s2limits,
        dataset=args.dataset,camind=args.camind,aif=args.aif,kcam=kcam,f=9.5e-3)
    print('s2 loss2: '+str(s2loss2))
    print('blur loss = '+str(blurloss))
    print('mean blur = '+str(meanblur))  
    print('min blur = '+str(minblur))
    print('max blur = '+str(maxblur)) 
    print('gt mean blur = '+str(gtmeanblur)) 
    print('__________________')
    
if __name__ == "__main__":
    main()

#plot MSE vs dist for various S1 values
import matplotlib.pyplot as plt
s2=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8]
mse1=[0.018,0.042,0.071,0.092,0.075,0.063,0.054,0.065,0.078,0.107,0.147,0.207,0.290,0.377,0.511,0.664,0.785,0.771]
mse2=[0.041,0.016,0.029,0.051,0.059,0.062,0.067,0.080,0.100,0.132,0.178,0.231,0.310,0.413,0.541,0.688,0.798,0.748]
mse3=[0.199,0.097,0.049,0.037,0.030,0.031,0.047,0.078,0.120,0.167,0.225,0.311,0.416,0.528,0.651,0.809,0.947,0.837]
mse4=[0.111,0.086,0.078,0.066,0.042,0.030,0.036,0.053,0.082,0.122,0.178,0.255,0.345,0.453,0.589,0.746,0.882,0.828]
mse5=[0.091,0.072,0.073,0.069,0.045,0.031,0.036,0.049,0.075,0.111,0.167,0.238,0.327,0.427,0.566,0.726,0.868,0.845]

plt.plot(s2,mse1,'-b',label='s1=0.1',marker=".", markersize=7)
plt.plot(s2,mse2,'-r',label='s1=0.15',marker="*", markersize=7)
plt.plot(s2,mse3,'-g',label='s1=0.3',marker="1", markersize=7)
plt.plot(s2,mse4,'-m',label='s1=0.7',marker="d", markersize=7)
plt.plot(s2,mse5,'-c',label='s1=1.5',marker="+", markersize=7)
plt.legend(loc="upper left")
plt.title('MSE vs distance')
plt.xlabel('distance(s2)-m')
plt.ylabel('MSE')
plt.show()






