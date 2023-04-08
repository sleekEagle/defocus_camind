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
parser.add_argument('--blenderpth', default="C://Users//lahir//focalstacks//datasets//mediumN1-10_test_remapped//", help='blender data path')
#parser.add_argument('--blenderpth', default="C:\\Users\\lahir\\focalstacks\\datasets\\medium_f_test\\", help='blender data path')
#parser.add_argument('--blenderpth', default="C://Users//lahir//focalstacks//datasets//mediumN1//", help='blender data path')
parser.add_argument('--kcamfile', default='kcams_est_DFV.txt', help='blender data path')
parser.add_argument('--ddffpth', default='C:\\Users\\lahir\\focalstacks\\datasets\\my_dff_trainVal.h5', help='blender data path')
parser.add_argument('--dataset', default='blender', help='blender data path')
parser.add_argument('--bs', type=int,default=1, help='training batch size')
parser.add_argument('--depthscale', default=1.9,help='divide all depths by this value')
parser.add_argument('--fscale', default=1.9,help='divide all focal distances by this value')
parser.add_argument('--savedmodel', default='C:\\Users\\lahir\\code\\defocus\\models\\defocus_trained\\camind.pth', help='path to the saved model')
#parser.add_argument('--savedmodel', default='C:\\Users\\lahir\\code\\defocus\\models\\a03_expcamind_norelu_N1_1.9_f1.9_blurclip8.0_blurweight1.0\\a03_expcamind_norelu_N1_1.9_f1.9_blurclip8.0_blurweight1.0_ep0.pth', help='path to the saved model')
#parser.add_argument('--savedmodel', default='C:\\Users\\lahir\\code\\defocus\\models\\a04_expaif_N1_d_1.9\\a04_expaif_N1_d_1.9_ep0.pth', help='path to the saved model')
#parser.add_argument('--savedmodel', default='C:\\Users\\lahir\\code\\defocus\\models\\a03_expcamind_fdistmul_N1_d_1.9_f1.9_blurclip8.0_blurweight0.3\\a03_expcamind_fdistmul_N1_d_1.9_f1.9_blurclip8.0_blurweight0.3_ep0.pth', help='path to the saved model')
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
        loaders, total_steps = focalblender.load_data(args.blenderpth,blur=1,aif=args.aif,train_split=1.0,fstack=0,WORKERS_NUM=0,
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
        s2loss1,s2loss2,blurloss,meanblur,gtmeanblur,minblur,maxblur=util_func.eval(loaders[0],model_info,args.depthscale,args.fscale,args.s2limits,
        dataset=args.dataset,camind=args.camind,aif=args.aif,calc_distmse=True)
        util_func.kcamwise_blur(loaders[0],model_info,args.depthscale,args.fscale,args.s2limits,camind=args.camind,aif=args.aif)
    elif(args.dataset=='ddff'):
        print('DDFF dataset Evaluation')
        kcam=5.0
        for kcam in [0.1,0.5,0.8,11,12,13,14,15,16,17,18]:
            print('kcam=%2.2f'%(kcam))
            s2loss1,s2loss2,blurloss,meanblur,gtmeanblur,minblur,maxblur=util_func.eval(TrainImgLoader,model_info,args.depthscale,args.fscale,args.s2limits,
            dataset=args.dataset,camind=args.camind,aif=args.aif,kcam=kcam,f=9.5e-3)
            print('MSE:%2.4f'%(s2loss2))

    print('s2 loss2: '+str(s2loss2))
    print('blur loss = '+str(blurloss))
    print('mean blur = '+str(meanblur))  
    print('min blur = '+str(minblur))
    print('max blur = '+str(maxblur)) 
    print('gt mean blur = '+str(gtmeanblur)) 
    print('__________________')
    
if __name__ == "__main__":
    main()

'''
#plot MSE vs dist for various S1 values
import matplotlib.pyplot as plt
s2=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8]
mse1=[0.026,0.040,0.053,0.058,0.067,0.087,0.121,0.171,0.239,0.319,0.422,0.538,0.685,0.855,1.040,1.253,1.489,1.709]
mse2=[0.050,0.028,0.026,0.032,0.045,0.067,0.105,0.157,0.229,0.312,0.418,0.541,0.690,0.857,1.049,1.270,1.505,1.707]
mse3=[0.126,0.089,0.049,0.022,0.013,0.018,0.040,0.077,0.133,0.199,0.291,0.404,0.534,0.686,0.861,1.064,1.288,1.545]
mse4=[0.114,0.099,0.073,0.041,0.023,0.018,0.028,0.054,0.099,0.155,0.235,0.336,0.453,0.592,0.750,0.941,1.153,1.458]
mse5=[0.128,0.114,0.092,0.060,0.035,0.024,0.026,0.043,0.078,0.126,0.197,0.288,0.393,0.519,0.667,0.842,1.044,1.384]

plt.plot(s2,mse1,'-b',label='s1=0.1',marker=".", markersize=7)
plt.plot(s2,mse2,'-r',label='s1=0.15',marker="*", markersize=7)
plt.plot(s2,mse3,'-g',label='s1=0.3',marker="1", markersize=7)
plt.plot(s2,mse4,'-m',label='s1=0.7',marker="d", markersize=7)
plt.plot(s2,mse5,'-c',label='s1=1.5',marker="+", markersize=7)
plt.legend(loc="upper left")
plt.title('MSE vs distance')
plt.xlabel('distance(s2)-m')
plt.ylabel('MSE')
plt.savefig('s2vsmse.png', dpi=500)
plt.show()
'''

'''
import util_func

p=3.1e-3/256
N=2
f=6e-3
s2range=[0.1,2.0]
s1range=[0.1,2.0]
blur_thres=3.0
util_func.get_workable_s1s2ranges(p,N,f,s2range,s1range,blur_thres,imgratio=1)
'''

'''
Evaluating focal length variation data
f=3mm
s1=1.5
s2 : 0.15 - 1.0
MSE
camind GTkcam:0.0516  kcamestGT:0.0500 kcamestDVF:0.0511
no camind:0.1005
defocus: 0.1932
aif:0.0996

f=4mm
s1=1.5
s2: 0.15-1.0
MSE
camind GTkcam:0.0478  kcamestGT:0.0422 kcamestDVF:0.0449
no camind: 0.0530
defocus: 0.1898
aif: 0.0819

f=5mm
s1=1.5
s2: 0.3-1.0
MSE
camind GTkcam:0.0547  kcamestGT:0.0488 kcamestDVF:0.0511
defocus : 0.1879
aif: 0.0827

f=6mm
s1=1.5
s2: 0.5 - 1.0
MSE
camind GTkcam:0.0620  kcamestGT:0.0604 kcamestDVF:0.0585
defocus : 0.0548
aif:0.1008
'''

'''
dist wise error 
s1=1.5m
with estimatef kcam
f=3mm
0.053,0.062,0.043,0.025,0.018,0.031,0.052,0.080,0.134,0.204,0.302,0.401,0.500,0.614,0.786,0.952,1.122,1.233
f=4mm
0.057,0.051,0.053,0.058,0.035,0.032,0.035,0.046,0.066,0.118,0.218,0.289,0.363,0.509,0.592,0.770,0.886,1.061
f=5mm
0.055,0.024,0.031,0.051,0.057,0.048,0.036,0.059,0.100,0.158,0.165,0.243,0.250,0.391,0.687,0.749,0.746,0.903
f=6mm
0.107,0.056,0.054,0.037,0.065,0.048,0.085,0.107,0.122,0.153,0.279,0.329,0.392,0.629,0.762,0.577,0.808,1.116
'''
'''
import matplotlib.pyplot as plt
s2=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8]
#DVF estimated kcam
mse1=[0.053,0.062,0.043,0.025,0.018,0.031,0.052,0.080,0.134,0.204,0.302,0.401,0.500,0.614,0.786,0.952,1.122,1.233]
mse2=[0.057,0.051,0.053,0.058,0.035,0.032,0.035,0.046,0.066,0.118,0.218,0.289,0.363,0.509,0.592,0.770,0.886,1.061]
mse3=[0.055,0.024,0.031,0.051,0.057,0.048,0.036,0.059,0.100,0.158,0.165,0.243,0.250,0.391,0.687,0.749,0.746,0.903]
mse4=[0.107,0.056,0.054,0.037,0.065,0.048,0.085,0.107,0.122,0.153,0.279,0.329,0.392,0.629,0.762,0.577,0.808,1.116]

#GT kcams
mse1=[0.185,0.102,0.081,0.055,0.084,0.044,0.102,0.107,0.096,0.131,0.228,0.290,0.347,0.553,0.730,0.527,0.850,0.815]
mse2=[0.068,0.031,0.031,0.054,0.066,0.050,0.037,0.055,0.097,0.143,0.164,0.256,0.259,0.412,0.709,0.750,0.749,0.842]
mse3=[0.069,0.052,0.057,0.060,0.041,0.033,0.033,0.043,0.063,0.109,0.209,0.279,0.360,0.500,0.580,0.732,0.856,1.009]
mse4=[0.061,0.071,0.051,0.032,0.022,0.028,0.042,0.066,0.111,0.172,0.266,0.361,0.456,0.554,0.723,0.874,1.021,1.118]

plt.plot(s2,mse1,'-b',label='f=3mm',marker=".", markersize=7)
plt.plot(s2,mse2,'-r',label='f=4mm',marker="*", markersize=7)
plt.plot(s2,mse3,'-g',label='f=5mm',marker="1", markersize=7)
plt.plot(s2,mse4,'-m',label='f=6mm',marker="d", markersize=7)
plt.legend(loc="upper left")
plt.title('MSE vs distance')
plt.xlabel('distance(s2)-m')
plt.ylabel('MSE')
plt.savefig('s2vsmse_f_DVFkcam.png', dpi=500)
plt.show()
'''
         0        1        2    3  --------4----------
                                     0     1     2 
list1=['apple','banana','mango',1,['car','van','ship']]

print(list1[4][1])









