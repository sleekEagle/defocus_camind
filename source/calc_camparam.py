#load the trained DFV model that is used to predict depth (s2)
import argparse
from DFV.models import DFFNet as DFFNet
import os
import time
from DFV.models.submodule import *
from torch.utils.data import DataLoader
from DFV.dataloader import FoD500Loader

import  matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from source import util_func

TRAIN_PARAMS = {
    'ARCH_NUM': 3,
    'FILTER_NUM': 16,
    'FLAG_GPU': True,
    'TRAINING_MODE': 2, #1: do not use step 1 , 2: use step 2
    'EPOCHS_NUM': 100, 'EPOCH_START': 0,
}

parser = argparse.ArgumentParser(description='defocu_camind')
parser.add_argument('--dfvmodel', default='C://Users//lahir//code//defocus//models//DFV//best.tar', help='DFV model path')
parser.add_argument('--camindmodel', default='C:\\Users\\lahir\\code\\defocus\\models\\a03_exp01\\a03_exp01_ep0.pth', help='DFV model path')
parser.add_argument('--data_path', default='C://Users//lahir//focalstacks//datasets//mediumN1//', help='data path to focal stacks')
args = parser.parse_args()

# construct DFV model and load weights
DFVmodel = DFFNet( clean=False,level=4, use_diff=1)
DFVmodel = nn.DataParallel(DFVmodel)
DFVmodel.cuda()

if args.dfvmodel is not None:
    pretrained_dict = torch.load(args.dfvmodel)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
    DFVmodel.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    print('run with random init')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in DFVmodel.parameters()])))


#get the single image depth prediction model
device_comp = util_func.set_comp_device(TRAIN_PARAMS['FLAG_GPU'])
model, inp_ch_num, out_ch_num = util_func.load_model(TRAIN_PARAMS)
model = model.to(device=device_comp)
model_params = model.parameters()
if args.camindmodel:
    print('loading model....')
    print('model path :'+args.camindmodel)
    pretrained_dict = torch.load(args.camindmodel)
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
                'model_params': model_params
                }

#dataloader
loaders, total_steps = util_func.load_data(args.data_path,blur=0,aif=0,train_split=0.8,fstack=1,WORKERS_NUM=0,
BATCH_SIZE=1,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,100000],REQ_F_IDX=[0,1,2,3,4],MAX_DPT=1.)
TrainImgLoader,ValImgLoader=loaders[0],loaders[1]


#calculate s2 for each focal stack
s2error=0
DFVmodel.eval()
for st_iter, sample_batch in enumerate(loaders[0]):

    #predicting s2
    img_stack=sample_batch['input'].float()
    gt_disp=sample_batch['output'][:,-1,:,:]
    gt_disp=torch.unsqueeze(gt_disp,dim=1).float()
    foc_dist=sample_batch['fdist'].float()

    img_stack_in   = Variable(torch.FloatTensor(img_stack))
    gt_disp    = Variable(torch.FloatTensor(gt_disp))
    img_stack, gt_disp, foc_dist = img_stack_in.cuda(),  gt_disp.cuda(), foc_dist.cuda()

    stacked, stds, _ = DFVmodel(img_stack, foc_dist)

    s2error+=torch.mean(torch.square(stacked-gt_disp)).detach().cpu().item()

    #calculate blur
    X = sample_batch['input'][:,:,:,:,:].float().to(device_comp)
    Y = sample_batch['output'].float().to(device_comp)

    gt_step1 = Y[:, :-1, :, :]
    gt_step2 = Y[:, -1:, :, :]
    
    stacknum = 5

    for t in range(stacknum):
        X2_fcs = torch.ones([X.shape[0],1, X.shape[3], X.shape[4]])
        s1_fcs = torch.ones([X.shape[0],1, X.shape[3], X.shape[4]])
        #iterate through the batch
        for i in range(X.shape[0]):
            focus_distance=sample_batch['fdist'][i][t].item()
            X2_fcs[i,0, :, :] = X2_fcs[i, 0, :, :]*(focus_distance-sample_batch['f'][i].item())/1.5*sample_batch['kcam'][i].item()/1.4398
            s1_fcs[i,0, :, :] = s1_fcs[i, 0, :, :]*(focus_distance)/1.5
        X2_fcs = X2_fcs.float().to(device_comp)
        s1_fcs = s1_fcs.float().to(device_comp)
        output_step1,output_step2 = util_func.forward_pass(X[:,t,:,:,:], model_info,stacknum=stacknum, additional_input=X2_fcs,foc_dist=s1_fcs)


print("MSE of depth prediction (s2) : "+str(s2error/len(loaders[0])))


















