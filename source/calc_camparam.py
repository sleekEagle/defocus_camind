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
import util_func


import sys
sys.path.append('../')

from source import util_func
from dataloaders import DDFF12,focalblender

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
parser.add_argument('--blenderpth', default='C:\\Users\\lahir\\focalstacks\\datasets\\mediumN1-3\\', help='blender data path')
parser.add_argument('--ddffpth', default='C:\\Users\\lahir\\focalstacks\\datasets\\my_dff_trainVal.h5', help='blender data path')
parser.add_argument('--dataset', default='ddff', help='DFV model path')
parser.add_argument('--s2limits', nargs='+', default=[0,1.0],  help='the interval of depth where the errors are calculated')
args = parser.parse_args()

# construct DFV model and load weights
DFVmodel = DFFNet(clean=False,level=4, use_diff=1)
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
if(args.dataset=='blender'):
    loaders, total_steps = focalblender.load_data(args.blenderpth,blur=1,aif=0,train_split=1.,fstack=1,WORKERS_NUM=0,
        BATCH_SIZE=1,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,100000],REQ_F_IDX=[0,1,2,3,4],MAX_DPT=1)
    TrainImgLoader,ValImgLoader=loaders[0],loaders[1]
if(args.dataset=='ddff'):
    DDFF12_train = DDFF12.DDFF12Loader(args.ddffpth, stack_key="stack_train", disp_key="disp_train", n_stack=10,
                                    min_disp=0.02, max_disp=0.28,fstack=1,idx_req=[9,8,0])
    DDFF12_val = DDFF12.DDFF12Loader(args.ddffpth, stack_key="stack_val", disp_key="disp_val", n_stack=10,
                                        min_disp=0.02, max_disp=0.28, b_test=False,fstack=1,idx_req=[9,8,0])
    DDFF12_train, DDFF12_val = [DDFF12_train], [DDFF12_val]

    dataset_train = torch.utils.data.ConcatDataset(DDFF12_train)
    dataset_val = torch.utils.data.ConcatDataset(DDFF12_val) # we use the model perform better on  DDFF12_val

    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=0, batch_size=1, shuffle=True, drop_last=True)
    ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val, num_workers=0, batch_size=1, shuffle=False, drop_last=True)
    f=3e-3


#calculate s2 for each focal stack
s2error,est_kcamlist,kcamlist=0,torch.empty(0),torch.empty(0)
DFVmodel.eval()
for st_iter, sample_batch in enumerate(TrainImgLoader):
    print(st_iter)
    if(args.dataset=='ddff'):
        img_stack, gt_disp, foc_dist=sample_batch
        foc_dist=foc_dist/1.9

        X=img_stack.float().to(device_comp)
        Y=gt_disp.float().to(device_comp)
        gt_step2=Y
    if(args.dataset=='blender'):
        X = sample_batch['input'][:,0,:,:,:].float().to(device_comp)
        Y = sample_batch['output'].float().to(device_comp)
        gt_step1 = Y[:, :-1, :, :]
        gt_step2 = Y[:, -1:, :, :]

        #preparing data for the DFV model to predict s2
        img_stack=sample_batch['input'].float()
        gt_disp=sample_batch['output'][:,-1,:,:]
        gt_disp=torch.unsqueeze(gt_disp,dim=1).float()
        foc_dist=sample_batch['fdist'].float()
        foc_dist=foc_dist/1.9

    img_stack_in   = Variable(torch.FloatTensor(img_stack))
    gt_disp    = Variable(torch.FloatTensor(gt_disp))
    img_stack, gt_disp, foc_dist = img_stack_in.cuda(),  gt_disp.cuda(), foc_dist.cuda()
    stacked, stds, _ = DFVmodel(img_stack, foc_dist)

    s2error+=torch.mean(torch.square(stacked-gt_disp)).detach().cpu().item()

    mask=(gt_step2>args.s2limits[0]).int()*(gt_step2<args.s2limits[1]).int()
    
    stacknum = X.shape[1]
    nandetected=False
    for t in range(stacknum):
        s1_fcs = torch.ones([X.shape[0],1, X.shape[3], X.shape[4]])
        s1f = torch.ones([X.shape[0],1, X.shape[3], X.shape[4]])
        #iterate through the batch
        for i in range(X.shape[0]):
            if(args.dataset=='blender'):
                focus_distance=sample_batch['fdist'][i][t].item()
                f=sample_batch['f'][i].item()
            if(args.dataset=='ddff'):
                focus_distance=foc_dist[i][t].item()
            s1_fcs[i,0, :, :] = s1_fcs[i, 0, :, :]*(focus_distance)
            s1f[i,0, :, :] = s1f[i, 0, :, :]*(focus_distance-f)
        s1_fcs = s1_fcs.float().to(device_comp)
        s1f = s1f.float().to(device_comp)

        blur_pred = util_func.forward_pass(X[:,t,:,:,:], model_info,stacknum=1,flag_step2=False)

        #calculate blur |s2-s1|/(s2*(s1-f)) from the DFV estimated s2
        est_kcam=torch.abs(stacked-s1_fcs)/stacked*1/(s1f)*1.4398/(10*blur_pred)*mask
        est_kcam=est_kcam[est_kcam>0]
        #remove outliers
        m=0.1
        d=torch.abs(est_kcam-torch.median(est_kcam))
        mdev=torch.median(d)
        s=d/mdev
        clean=est_kcam[s<m]
        #if there are no items meeting the criteria, continue.
        if(len(clean)==0):
            continue
        est_kcamlist=torch.cat((est_kcamlist,torch.mean(clean).detach().cpu().unsqueeze(dim=0)))
        if(torch.sum(torch.isnan(est_kcamlist)).item()>0):
            print('there is nan. breaking')
            nandetected=True
            break
        if(args.dataset=='blender'):
            kcamlist=torch.cat((kcamlist,sample_batch['kcam'].detach().cpu()))
    print("isnan?"+str(torch.sum(torch.isnan(est_kcamlist)).item()))
    if(nandetected):
        break

#get camera-wise kcam estimation
def reject_outliers(data, m = 1):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zero(len(d))
    return data[s<m]

if(args.dataset=='blender'):
    unique_kcams, _ = kcamlist.unique(dim=0, return_counts=True)
    kcam_est_list=torch.empty((0))
    for i in range(unique_kcams.shape[0]):
        indices=((kcamlist == unique_kcams[i].item()).nonzero(as_tuple=True)[0])
        estkcams_i=est_kcamlist[indices]
        #remove outliers (yes, again)
        estkcams_i_clean=reject_outliers(estkcams_i,m=0.1)
        kcam_est=torch.mean(estkcams_i_clean).unsqueeze(dim=0)
        kcam_est_list=torch.cat((kcam_est_list,kcam_est))

    errors=((kcam_est_list-unique_kcams)/unique_kcams).numpy()

    plt.scatter(unique_kcams,abs(errors))
    plt.title('Depth range 0.1-1.0 m')
    plt.xlabel('Kcam')
    plt.ylabel('MSE of Kcam estimation')
    plt.show()

if(args.dataset=='ddff'):
    est_kcamlist_list=reject_outliers(est_kcamlist,m=0.1)
    print("Estimated Kcam = "+str(torch.mean(est_kcamlist_list).item()))


