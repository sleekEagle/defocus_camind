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
parser.add_argument('--camindmodel', default='C:\\Users\\lahir\\code\\defocus\\models\\a03_exp01\\a03_exp01_ep0.pth', help='DFV model path')
parser.add_argument('--blenderpth', default='C:\\Users\\lahir\\focalstacks\\datasets\\mediumN1\\', help='blender data path')
parser.add_argument('--ddffpth', default='C:\\Users\\lahir\\focalstacks\\datasets\\my_dff_trainVal.h5', help='blender data path')
parser.add_argument('--dataset', default='blender', help='DFV model path')
parser.add_argument('--s2limits', nargs='+', default=[0.01,1.0],  help='the interval of depth where the errors are calculated')
parser.add_argument('--depthscale', default=1.9,help='divide all depths by this value')
parser.add_argument('--fscale', default=1.9,help='divide all focal distances by this value')
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
        BATCH_SIZE=1,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,100000],REQ_F_IDX=[0,1,2,3,4],MAX_DPT=1)
    TrainImgLoader,ValImgLoader=loaders[0],loaders[1]
if(args.dataset=='ddff'):
    DDFF12_train = DDFF12.DDFF12Loader(args.ddffpth, stack_key="stack_train", disp_key="disp_train", n_stack=10,
                                    min_disp=0.02, max_disp=0.28,fstack=0,idx_req=[7,8,9])
    DDFF12_val = DDFF12.DDFF12Loader(args.ddffpth, stack_key="stack_val", disp_key="disp_val", n_stack=10,
                                        min_disp=0.02, max_disp=0.28, b_test=False,fstack=0,idx_req=[6,5,4,3,2,1,0])
    DDFF12_train, DDFF12_val = [DDFF12_train], [DDFF12_val]

    dataset_train = torch.utils.data.ConcatDataset(DDFF12_train)
    dataset_val = torch.utils.data.ConcatDataset(DDFF12_val) # we use the model perform better on  DDFF12_val

    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=0, batch_size=1, shuffle=True, drop_last=True)
    ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val, num_workers=0, batch_size=1, shuffle=False, drop_last=True)
    f=3e-3

#calculate s2 for each focal stack
def method1():
    s2error,est_kcamlist,kcamlist=0,torch.empty(0),torch.empty(0)
    ulist,vlist=torch.empty(0),torch.empty(0)
    for st_iter, sample_batch in enumerate(TrainImgLoader):
        print(st_iter)
        if(st_iter>10):
            break
        if(args.dataset=='ddff'):
            img_stack, gt_disp, foc_dist=sample_batch
            X=img_stack.float().to(device_comp)
            Y=gt_disp.float().to(device_comp)
            gt_step2=Y
        if(args.dataset=='blender'):
            X = sample_batch['input'].float().to(device_comp)
            Y = sample_batch['output'].float().to(device_comp)
            gt_step1 = Y[:, :-1, :, :]
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

        print('mask '+str(mask.shape))

        stacknum = X.shape[1]
        bs=X.shape[0]
        #iterate though the batch
        s1_fcs = torch.ones([bs, stacknum, X.shape[3], X.shape[4]])
        blur_preds = torch.ones([bs, stacknum, X.shape[3], X.shape[4]])
        lmd=torch.ones([bs, stacknum, X.shape[3], X.shape[4]])
        for i in range(bs):
            #iterate through the focal stack
            for t in range(stacknum):
                if(args.dataset=='ddff'):
                    focus_distance=foc_dist[i][t].item()
                if(args.dataset=='blender'):
                    focus_distance=sample_batch['fdist'][i][t].item()                
                s1_fcs[i,t, :, :] = s1_fcs[i,t, :, :]*(focus_distance)
                s1_fcs = s1_fcs.float().to(device_comp)
                img=torch.unsqueeze(X[i,t,:,:,:],dim=0)
                blur_pred = util_func.forward_pass(img, model_info,stacknum=1,flag_step2=False)
                blur_preds[i,t,:,:]=blur_pred*10/1.4398
                lmd[i,t,:,:]=torch.abs(gt_step2-s1_fcs[i,t,:,:])/(gt_step2)


        lmd[0,0,200,100]

        torch.abs(gt_step2[0,0,200,100]-s1_fcs[0,0,200,100])/(gt_step2[0,0,200,100])

        b1=blur_preds[0,0,200,100]
        bi=blur_preds[0,1,200,100]
        l1=lmd[0,0,200,100]
        li=lmd[0,1,200,100]
        s11=s1_fcs[0,0,200,100]
        s1i=s1_fcs[0,1,200,100]

        f=(b1/bi*s11 - l1/li*s1i)/(b1/b1-l1/li)


        lmd_last=lmd[:,-1,:,:].unsqueeze(dim=1)
        lmd=lmd/lmd_last
        blur_preds_last=blur_preds[:,-1,:,:].unsqueeze(dim=1)
        blur_preds=blur_preds/blur_preds_last

        s1=s1_fcs.detach().cpu()
        s1_last=torch.repeat_interleave(torch.unsqueeze(s1[:,-1,:,:],dim=1),repeats=lmd.shape[1],dim=1)
        f=(blur_preds*s1 - lmd*s1_last)/(blur_preds-lmd)
        f=f[:,0:-4,:,:]*torch.repeat_interleave(mask,repeats=4,dim=1).cpu()
        f=f[f>0]
        f=f[f<10]
        torch.mean(f)


                #we cannot estimate kcam or f when there is no blur
                v=s1_fcs[mask>0].detach().cpu()
                u=(torch.abs(gt_step2-s1_fcs)/(gt_step2)*1.4398/(10*blur_pred))[mask>0].detach().cpu()
                kcamest=

                #torch.mean(u[u>0]/v[v>0])

                vlist=torch.cat((vlist,v[v>0]))
                ulist=torch.cat((ulist,u[u>0]))

                torch.mean(ulist/vlist)
                vlistnp=np.expand_dims(vlist.numpy(),axis=1)
                ulistnp=np.expand_dims(ulist.numpy(),axis=1)

                _ = plt.hist(vlistnp, bins='auto')
                plt.show()

                np.mean((ulistnp-3e-3)/vlistnp)

                import matplotlib.pyplot as plt
                plt.scatter(vlistnp,ulistnp)
                plt.show()
                from sklearn import linear_model, datasets

                ransac = linear_model.RANSACRegressor(min_samples=10000)
                ransac.fit(ulistnp,vlistnp)
                print(1/ransac.estimator_.coef_)
                ransac.estimator_.intercept_

                lr = linear_model.LinearRegression()
                lr.fit(ulistnp,vlistnp)
                print(1/lr.coef_)

                plt.scatter(vlistnp,ulistnp)
                plt.show()

                slope, intercept, r_value, p_value, std_err = stats.linregress(ulistnp,vlistnp)
                1/slope
            

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
                if(args.dataset=='blender'):
                    kcamlist=torch.cat((kcamlist,sample_batch['kcam'].detach().cpu()))

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

#grid search based method to find both kcam and f
#this takes a longer time due to grid-search
s2error,est_kcamlist,kcamlist=0,torch.empty(0),torch.empty(0)
count=0
kcam=4.7073
f=3e-3

s2loss1,s2loss2,blurloss,meanblur=util_func.eval(TrainImgLoader,model_info,args.depthscale,args.fscale,args.s2limits,
                                dataset=args.dataset,kcam=15,f=0.079)   

for st_iter, sample_batch in enumerate(TrainImgLoader):
    if(st_iter==10):
        break
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

    mask=(gt_step2>args.s2limits[0]).int()*(gt_step2<args.s2limits[1]).int()

    stacknum = X.shape[1]
    bs=X.shape[0]
    #iterate though the batch
    for i in range(bs):
        #iterate through the focal stack
        for t in range(stacknum):
            X2_fcs = torch.ones([1, 1, X.shape[3], X.shape[4]])
            s1_fcs = torch.ones([1, 1, X.shape[3], X.shape[4]])
            if(args.dataset=='blender'):
                focus_distance=sample_batch['fdist'][i][t].item()
            elif(args.dataset=='ddff'):
                focus_distance=foc_dist[i][t].item()
            X2_fcs[:, :, :, :] = X2_fcs[:,:, :, :]*(focus_distance-f)/1.9*kcam/1.4398
            s1_fcs[:,:, :, :] = s1_fcs[:,:, :, :]*(focus_distance)/1.9
            X2_fcs = X2_fcs.float().to(model_info['device_comp'])
            s1_fcs = s1_fcs.float().to(model_info['device_comp'])
            #predict depth whith camind model
            img=torch.unsqueeze(X[i,t,:,:,:],dim=0)
            output_step1,output_step2 = util_func.forward_pass(img, model_info,stacknum=1, additional_input=X2_fcs,foc_dist=s1_fcs)
            s2error+=torch.mean(torch.square(output_step2*1.9-gt_step2)*mask).detach().cpu().item()
            count+=1
s2error/count
    

def main():
    method1()

if __name__ == "__main__":
    main()