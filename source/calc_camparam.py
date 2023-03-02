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
from scipy import stats


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
parser.add_argument('--camindmodel', default='C:\\Users\\lahir\\code\\defocus\\models\\a03_expcamind_fdistmul_N1_d_1.9_f1.9_blurclip8.0_blurweight0.3\\a03_expcamind_fdistmul_N1_d_1.9_f1.9_blurclip8.0_blurweight0.3_ep0.pth', help='camind model path')
parser.add_argument('--blenderpth', default='C:\\Users\\lahir\\focalstacks\\datasets\\mediumN1-10_test_remapped\\', help='blender data path')
parser.add_argument('--ddffpth', default='C:\\Users\\lahir\\focalstacks\\datasets\\my_dff_trainVal.h5', help='blender data path')
parser.add_argument('--dataset', default='blender', help='DFV model path')
parser.add_argument('--s2limits', nargs='+', default=[0.1,0.3],  help='the interval of depth where the errors are calculated')
parser.add_argument('--depthscale', default=1.9,help='divide all depths by this value')
parser.add_argument('--fscale', default=1.9,help='divide all focal distances by this value')
parser.add_argument('--blurnorm', type=int,default=8.0, help='blur normalization value used to train camind model (all blur values were divided by this value)')
parser.add_argument('--usegt', type=bool,default=False, help='True: use GT depth values when estimation kcams False: use DFV estimated depth values instead (more realistic)')
parser.add_argument('--maximgs', type=int,default=5, help='max number of focal stacks (per one camera) used to estimate kcams')
parser.add_argument('--s1indices', nargs='+', default=[0,1,2,3,4], help='indices of focal distances used to estimate kcam')
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
DFVmodel.eval()

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
        BATCH_SIZE=1,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,100000],REQ_F_IDX=args.s1indices,MAX_DPT=1)
    TrainImgLoader,ValImgLoader=loaders[0],loaders[1]
if(args.dataset=='ddff'):
    DDFF12_train = DDFF12.DDFF12Loader(args.ddffpth, stack_key="stack_train", disp_key="disp_train", n_stack=10,
                                    min_disp=0.02, max_disp=0.28,fstack=1,idx_req=args.s1indices)
    DDFF12_val = DDFF12.DDFF12Loader(args.ddffpth, stack_key="stack_val", disp_key="disp_val", n_stack=10,
                                        min_disp=0.02, max_disp=0.28, b_test=False,fstack=0,idx_req=[6,5,4,3,2,1,0])
    DDFF12_train, DDFF12_val = [DDFF12_train], [DDFF12_val]

    dataset_train = torch.utils.data.ConcatDataset(DDFF12_train)
    dataset_val = torch.utils.data.ConcatDataset(DDFF12_val) # we use the model perform better on  DDFF12_val

    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=0, batch_size=1, shuffle=True, drop_last=True)
    ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val, num_workers=0, batch_size=1, shuffle=False, drop_last=True)
#calculate s2 for each focal stack
def est_kcam(f,maximgs=20,usegt=False):
    s2error,est_kcamlist,kcamlist=0,torch.empty(0),torch.empty(0)
    meanf=0
    count=0
    for st_iter, sample_batch in enumerate(TrainImgLoader):
        #if(st_iter>200):
        #    break
        #break if we have the minimum requred number of images (maximgs) from each kcam
        unique_kcams, _ = kcamlist.unique(dim=0, return_counts=True)
        minn=100
        for k in unique_kcams:
            n=len(kcamlist[kcamlist==k])
            if n<minn:
                minn=n
        if(st_iter>0 and minn>=maximgs):
            break
        if(args.dataset=='ddff'):
            img_stack, gt_disp, foc_dist=sample_batch
            X=img_stack.float().to(device_comp)
            Y=gt_disp.float().to(device_comp)
            gt_step2=Y
        if(args.dataset=='blender'):
            X = sample_batch['input'].float().to(device_comp)
            Y = sample_batch['output'].float().to(device_comp)
            gt_step2 = Y[:, -1:, :, :]

            #preparing data for the DFV model to predict s2
            img_stack=sample_batch['input'].float()
            gt_disp=sample_batch['output'][:,-1,:,:]
            gt_disp=torch.unsqueeze(gt_disp,dim=1).float()
            foc_dist=sample_batch['fdist'].float()

        mask=(gt_step2>args.s2limits[0]).int()*(gt_step2<args.s2limits[1]).int()

        img_stack_in   = Variable(torch.FloatTensor(img_stack))
        gt_disp    = Variable(torch.FloatTensor(gt_disp))
        img_stack, gt_disp, foc_dist = img_stack_in.cuda(),  gt_disp.cuda(), foc_dist.cuda()
        stacked, stds, _ = DFVmodel(img_stack, foc_dist/args.fscale)

        s2error+=torch.mean(torch.square(stacked*args.depthscale-gt_disp)*mask).detach().cpu().item()

        stacknum = X.shape[1]
        bs=X.shape[0]
        #iterate though the batch
        s1=torch.ones([bs, stacknum, X.shape[3], X.shape[4]])
        s1f=torch.ones([bs, stacknum, X.shape[3], X.shape[4]])
        blur_preds = torch.ones([bs, stacknum, X.shape[3], X.shape[4]])
        stacked=stacked.to("cpu")
        for i in range(bs):
            #if we have this kam the required number of times, continue
            if(args.dataset=='blender'):
                kcam=sample_batch['kcam'].detach().cpu()
                n=len(kcamlist[kcamlist==kcam])
                if(n>=maximgs):
                    continue
            #iterate through the focal stack
            for t in range(stacknum):
                if(args.dataset=='ddff'):
                    focus_distance=foc_dist[i][t].item()
                if(args.dataset=='blender'):
                    focus_distance=sample_batch['fdist'][i][t].item()                
                s1[i,t, :, :] = s1[i,t, :, :]*(focus_distance)
                s1f[i,t, :, :] = s1f[i,t, :, :]*(focus_distance-f)
                img=torch.unsqueeze(X[i,t,:,:,:],dim=0)
                blur_pred,mul = util_func.forward_pass(img, model_info,stacknum=1,flag_step2=False)
                blur_preds[i,t,:,:]=blur_pred

                s1f_=s1f[i,t, :, :].unsqueeze(dim=0).unsqueeze(dim=1)
                s1_=s1[i,t, :, :].unsqueeze(dim=0).unsqueeze(dim=1)
                if(usegt):
                    est_kcam=torch.abs(gt_step2.cpu()-s1_)/(gt_step2.cpu())*1/(s1f_)/(args.blurnorm*blur_pred.cpu())*mask.cpu()
                else:
                    est_kcam=torch.abs(stacked.cpu()-s1_)/(stacked.cpu())*1/(s1f_)/(args.blurnorm*blur_pred.cpu())*mask.cpu()
                est_kcam=est_kcam[est_kcam>0]
                est_kcamlist=torch.cat((est_kcamlist,torch.mean(est_kcam).detach().cpu().unsqueeze(dim=0)))
                if(args.dataset=='blender'):
                    kcamlist=torch.cat((kcamlist,sample_batch['kcam'].detach().cpu()))

        #estimating f
        '''
        for i in range(stacknum):
            for j in range(stacknum):
                if(i==j):
                    continue
                s1_1=s1[:,i,:,:].unsqueeze(dim=1)
                blur_=blur_preds[:,i,:,:].unsqueeze(dim=1)
                bigmask=(blur_>0.01).int()
                l1=torch.abs(gt_step2.cpu()-s1_1)/(gt_step2.cpu())/(blur_)
                bigmask*=(torch.abs(gt_step2.cpu()-s1_1)>0.01).int()
                s1_2=s1[:,j,:,:].unsqueeze(dim=1)
                blur_=blur_preds[:,j,:,:].unsqueeze(dim=1)
                bigmask*=(blur_>0.01).int()
                l2=torch.abs(gt_step2.cpu()-s1_2)/(gt_step2.cpu())/(blur_)
                bigmask*=(torch.abs(gt_step2.cpu()-s1_2)>0.01).int()

                m=l1/l2

                f_est=(s1_1-m*s1_2)/(1-m)*mask.cpu()*bigmask
                meanf+=torch.mean(f_est[f_est>0]).item()
                count+=1
    print('estimated f = '+str(meanf/count))
    '''

    #estimating Kcam
    def reject_outliers(data, m = 1):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else np.zero(len(d))
        return data[s<m]

    if(args.dataset=='blender'):
        unique_kcams, nums = kcamlist.unique(dim=0, return_counts=True)
        kcam_est_list=torch.empty((0))
        for i in range(unique_kcams.shape[0]):
            indices=((kcamlist == unique_kcams[i].item()).nonzero(as_tuple=True)[0])
            estkcams_i=est_kcamlist[indices]
            #remove outliers (yes, again)
            estkcams_i_clean=reject_outliers(estkcams_i,m=0.5)
            kcam_est=torch.mean(estkcams_i_clean).unsqueeze(dim=0)
            kcam_est_list=torch.cat((kcam_est_list,kcam_est))

        errors=((kcam_est_list-unique_kcams)/unique_kcams).numpy()

        plt.scatter(unique_kcams,abs(errors))
        plt.title('MSE of Kcam estimation')
        plt.xlabel('Kcam')
        plt.ylabel('MSE')
        plt.show()
        print('real kcams: '+str(unique_kcams.tolist()))
        print('num images : '+str(nums))
        print('estimated kcams: '+str(kcam_est_list.tolist()))
        print('s2 estimation error: '+str(s2error/len(TrainImgLoader)))

    if(args.dataset=='ddff'):
        est_kcamlist_list=reject_outliers(est_kcamlist,m=0.1)
        print("Estimated Kcam = "+str(torch.mean(est_kcamlist_list).item()))

def main():
    if(args.dataset=='blender'):
        f=2.9e-3
    if(args.dataset=='ddff'):
        f=9.5e-3
    #need to mutiply maximgs by len(args.s1indices) because maximgs is the number of focal stacks used
    #but we need the number of images
    est_kcam(f,args.maximgs*len(args.s1indices),args.usegt)

if __name__ == "__main__":
    main()