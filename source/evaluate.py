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
parser.add_argument('--blenderpth', default="C:\\Users\\lahir\\focalstacks\\datasets\\mediumN1-10_test_remapped\\", help='blender data path')
parser.add_argument('--kcamfile', default="kcams_gt.txt", help='blender data path')
parser.add_argument('--ddffpth', default='C:\\Users\\lahir\\focalstacks\\datasets\\my_dff_trainVal.h5', help='blender data path')
parser.add_argument('--dataset', default='blender', help='blender data path')
parser.add_argument('--bs', type=int,default=1, help='training batch size')
parser.add_argument('--depthscale', default=1.9,help='divide all depths by this value')
parser.add_argument('--fscale', default=1.9,help='divide all focal distances by this value')
#parser.add_argument('--savedmodel', default='C:\\Users\\lahir\\code\\defocus\\models1\\a03_expdefocus_d1.9_f1.9\\a03_expdefocus_d1.9_f1.9_ep0.pth', help='path to the saved model')
parser.add_argument('--savedmodel', default='C:\\Users\\lahir\\code\\defocus\\models\\a04_expaif_N1_d_1.9\\a04_expaif_N1_d_1.9_ep0.pth', help='path to the saved model')
parser.add_argument('--s2limits', nargs='+', default=[0.1,1.0],  help='the interval of depth where the errors are calculated')
parser.add_argument('--camind', type=bool,default=True, help='True: use camera independent model. False: use defocusnet model')
parser.add_argument('--aif', type=bool,default=True, help='True: Train with the AiF images. False: Train with blurred images')
args = parser.parse_args()

if(args.aif):
    TRAIN_PARAMS['ARCH_NUM']=4

def main():
    device_comp = util_func.set_comp_device(TRAIN_PARAMS['FLAG_GPU'])
    #load the required dataset
    if(args.dataset=='blender'):
        loaders, total_steps = focalblender.load_data(args.blenderpth,blur=1,aif=args.aif,train_split=1,fstack=0,WORKERS_NUM=0,
        BATCH_SIZE=args.bs,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,100000],REQ_F_IDX=[0,1,2,3,4],MAX_DPT=1,kcampath=args.blenderpth+args.kcamfile)
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
        dataset=args.dataset,camind=args.camind,aif=args.aif)
        util_func.kcamwise_blur(loaders[0],model_info,args.depthscale,args.fscale,args.s2limits,camind=args.camind,aif=args.aif)
    elif(args.dataset=='ddff'):
        print('DDFF dataset Evaluation')
        kcam=3
        s2loss1,s2loss2,blurloss,meanblur,gtmeanblur,minblur,maxblur=util_func.eval(TrainImgLoader,model_info,args.depthscale,args.fscale,args.s2limits,
        dataset=args.dataset,camind=args.camind,kcam=kcam,f=9.5e-3)
    print('s2 loss2: '+str(s2loss2))
    print('blur loss = '+str(blurloss))
    print('mean blur = '+str(meanblur))  
    print('min blur = '+str(minblur))
    print('max blur = '+str(maxblur)) 
    print('gt mean blur = '+str(gtmeanblur)) 
    print('__________________')

'''
import matplotlib.pyplot as plt

gt=torch.unsqueeze(gt_blur,dim=0)
pred=torch.unsqueeze(pred_blur,dim=0)
error=pred_blur-gt_blur


blur_l,error_l=[],[]
r=np.arange(0,0.6,0.001)
for i in range(len(r)-1):
    blur_l.append(r[i])
    error_l.append(torch.mean(error[(gt_blur>r[i])*(gt_blur<r[i+1])]).item())

plt.scatter(blur_l,error_l)
plt.show()



torch.mean(error[gt_blur<0.003])
torch.mean(error[(gt_blur>0.003)*(gt_blur<0.03)])
torch.mean(error[(gt_blur>0.03)*(gt_blur<0.1)])
torch.mean(error[(gt_blur>0.03)*(gt_blur<0.1)])
torch.mean(error[(gt_blur>0.1)*(gt_blur<0.2)])
torch.mean(error[(gt_blur>0.2)*(gt_blur<0.3)])
torch.mean(error[(gt_blur>0.3)*(gt_blur<0.4)])
torch.mean(error[(gt_blur>0.4)*(gt_blur<0.6)])
'''



         
     
if __name__ == "__main__":
    main()

'''
import numpy as np
import matplotlib.pyplot as plt
import math
k=np.arange(1,100)
vals=np.ones_like(k)/k

corrected=vals*k
plt.plot(vals)
plt.plot(corrected)
plt.show()



vals=np.array([0.2727, 0.2707, 0.2496, 0.2067, 0.1554, 0.1453, 0.1331, 0.1182, 0.1150,0.0991, 0.0955, 0.0965])
k=np.array([1.4399,  1.5839,  1.7279,  2.1598,  2.5918,  2.8798,  3.1677,  4.0317,4.3196,  7.1994, 11.5190, 14.3988])
corrected=vals/1.4399*k*(0.9**k)
plt.scatter(k,vals)
plt.scatter(k,corrected)
plt.show()
'''



'''  
import numpy as np

f=9.5e-3
blurs=[]
s2list,s1list=[],[]
for s2 in np.linspace(0.02,0.28,10):
    for s1 in np.linspace(0.02,0.28,10):
        blur=abs(s2-s1)/s2*1/(s1-f)
        blurs.append(blur)
        s1list.append(s1)
        s2list.append(s2)
print('min: '+str(min(blurs))+' max='+str(max(blurs))) 
print('min s2 :'+str(s2list[np.argmin(np.array(blurs))])+' max s2 :'+str(s2list[np.argmax(np.array(blurs))]))
print('min s1 :'+str(s1list[np.argmin(np.array(blurs))])+' max s1 :'+str(s1list[np.argmax(np.array(blurs))]))


f=2.9e-3
blurs=[]
s2list,s1list=[],[]
for s2 in np.linspace(0.02,1.89,10):
    for s1 in np.linspace(0.1,1.5,10):
        blur=abs(s2-s1)/s2*1/(s1-f)
        blurs.append(blur)
        s1list.append(s1)
        s2list.append(s2)
print('min blur: '+str(min(blurs))+' max blur='+str(max(blurs))) 
print('min s2 :'+str(s2list[np.argmin(np.array(blurs))])+' max s2 :'+str(s2list[np.argmax(np.array(blurs))]))
print('min s1 :'+str(s1list[np.argmin(np.array(blurs))])+' max s1 :'+str(s1list[np.argmax(np.array(blurs))]))
'''
