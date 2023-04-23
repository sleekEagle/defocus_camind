import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import math
import argparse
from dataloaders import focalblender,NUY_blurred
from arch import dofNet_arch3
import sys
import os


parser = argparse.ArgumentParser(description='camIndDefocus')
# parser.add_argument('--datapath', default='C:\\Users\\lahir\\data\\nyu_depth\\noborders\\', help='blender data path')
parser.add_argument('--datapath', default='C:\\Users\\lahir\\focalstacks\\datasets\\defocusnet_N1\\', help='blender data path')
parser.add_argument('--bs', type=int,default=20, help='training batch size')
parser.add_argument('--epochs', type=int,default=1000, help='training batch size')
parser.add_argument('--depthscale', default=15.,help='divide all depths by this value')
parser.add_argument('--blurclip', default=6.5,help='Clip blur by this value : only applicable for camind model. Default=10')
parser.add_argument('--blurweight', default=1.0,help='weight for blur loss')
parser.add_argument('--savepath', default='C:\\Users\\lahir\\code\\defocus\\models\\', help='path to the saved model')
parser.add_argument('--checkpt', default='C:\\Users\\lahir\\code\\defocus\\models\\camind_defocusnet_15.0_blurclip6.5_blurweight1.0\\model.pth', help='path to the saved model')
# parser.add_argument('--checkpt', default=None, help='path to the saved model')
parser.add_argument('--s2limits', nargs='+', default=[0.1,1.5],  help='the interval of depth where the errors are calculated')
parser.add_argument('--dataset', default='defocusnet', help='blender data path')
parser.add_argument('--camind', type=bool,default=True, help='True: use camera independent model. False: use defocusnet model')
parser.add_argument('--aif', type=bool,default=False, help='True: Train with the AiF images. False: Train with blurred images')
parser.add_argument('--out_depth', type=bool,default=False, help='True: use camera independent model. False: use defocusnet model')
parser.add_argument('--lr', default=0.00001,help='dilvide all depths by this value')
args = parser.parse_args()

if(args.aif):   
    expname='aif_defnetdata_'+str(args.depthscale)
else:
    if(args.camind):
        expname='camind_'+str(args.dataset)+'_'+str(args.depthscale)+'_blurclip'+str(args.blurclip)+'_blurweight'+str(args.blurweight)
    else:
        expname='nocamind_'+str(args.dataset)+'_'+str(args.depthscale)+'_blurclip'+str(args.blurclip)+'_blurweight'+str(args.blurweight)
#create directory to save model
if not os.path.exists(args.savepath +expname):
    os.makedirs(args.savepath +expname)

'''
load model
'''
#GPU or CPU
device_comp = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ch_inp_num = 3
ch_out_num = 1
model = dofNet_arch3.AENet(ch_inp_num, 1, 16, flag_step2=True)
model = model.to(device_comp)
model_params = model.parameters()

# loading weights of the first step
if args.checkpt:
    print('loading model....')
    print('model path :'+args.checkpt)
    pretrained_dict = torch.load(args.checkpt)
    model_dict = model.state_dict()
    for param_tensor in model_dict:
        for param_pre in pretrained_dict:
            if param_tensor == param_pre:
                model_dict.update({param_tensor: pretrained_dict[param_pre]})
    model.load_state_dict(model_dict)

'''
Load data
'''
 # Training initializations
if(args.dataset=='blender'):
    loaders, total_steps = focalblender.load_data(args.datapath,blur=1,aif=args.aif,train_split=0.8,fstack=0,WORKERS_NUM=0,
    BATCH_SIZE=args.bs,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,100000],REQ_F_IDX=[0,1,2,3,4],MAX_DPT=1.0,
    blurclip=args.blurclip,dataset=args.dataset)
elif(args.dataset=='defocusnet'):
    loaders, total_steps = focalblender.load_data(args.datapath,blur=1,aif=0,train_split=0.8,fstack=0,WORKERS_NUM=0,
    BATCH_SIZE=args.bs,FOCUS_DIST=[0.1,.15,.3,0.7,1.5],REQ_F_IDX=[0,1,2,3,4],MAX_DPT=1.0,blurclip=1.0,dataset=args.dataset,
    out_depth=args.out_depth)
elif(args.dataset=='nyu'):
    rgbpath=args.datapath+"refocused1\\"
    depthpath=args.datapath+"depth\\"
    kcampath=args.datapath+"refocused1\\camparam.txt"
    loaders, total_steps = NUY_blurred.load_data(rgbpath=rgbpath,depthpath=depthpath,blur=1,train_split=0.8,fstack=0,WORKERS_NUM=0,
            BATCH_SIZE=20,MAX_DPT=1,blurclip=args.blurclip,kcampath=kcampath)


# ============ init ===============
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)

NORM_MIN=0.05
NORM_MAX=15.0
def normalize(x):
    v=(x-NORM_MIN)/(NORM_MAX-NORM_MIN)
    return v
def denormalize(v):
    x=v*(NORM_MAX-NORM_MIN)+NORM_MIN
    return x

def eval(loader,kcam=0,f=0,calc_distmse=False,out_depth=False):
    meanMSE,meanMSE2,meanblurmse,meanblur=0,0,0,0
    minblur,maxblur,gt_meanblur=100,0,0
    #store distance wise mse
    distmse,distsum,distblur=torch.zeros(100),torch.zeros(100),torch.zeros(100)

    print('Total samples = '+str(len(loader)))
    for st_iter, sample_batch in enumerate(loader):
        #if(st_iter>100):
        #    break
        #sys.stdout.write(str(st_iter)+" of "+str(len(loader))+" is done")
        sys.stdout.write("\r%d is done"%st_iter)
        sys.stdout.flush()

        if(args.dataset=='ddff'):
            img_stack, gt_disp, foc_dist=sample_batch
            X=img_stack.float().to(device_comp)
            Y=gt_disp.float().to(device_comp)
            gt_step2=Y
        elif(args.dataset=='blender' or args.dataset=='defocusnet'):
            # Setting up input and output data
            X = sample_batch['input'][:,0,:,:,:].float().to(device_comp)
            depth=sample_batch['depth'].float().to(device_comp)
            blur=sample_batch['blur'].float().to(device_comp)
            focus_distance=sample_batch['fdist']
            focus_distance=torch.unsqueeze(focus_distance,dim=2).unsqueeze(dim=3)
            focus_distance=torch.repeat_interleave(focus_distance,depth.shape[2],dim=2).repeat_interleave(depth.shape[3],dim=3)
            focus_distance=focus_distance.to(device_comp)
        elif(args.dataset=="nyu"):
            X=sample_batch['rgb'].float().to(device_comp)
            depth=sample_batch['depth'].float().to(device_comp)
            blur=sample_batch['blur'].float().to(device_comp)
            depth=torch.unsqueeze(depth,dim=1)
            depth=torch.unsqueeze(depth,dim=1)
            focus_distance=sample_batch['fdist']
            focus_distance=torch.unsqueeze(focus_distance,dim=2).unsqueeze(dim=3)
            focus_distance=torch.repeat_interleave(focus_distance,depth.shape[2],dim=2).repeat_interleave(depth.shape[3],dim=3)
            focus_distance=focus_distance.to(device_comp)

        if(len(args.s2limits)==2):
            mask=(focus_distance/depth>args.s2limits[0])*(focus_distance/depth<args.s2limits[1])
            s=torch.sum(mask).item()
            #continue loop if there are no ground truth data in the range we are interested in
            if(s==0):
                continue
        else:
            mask=torch.ones_like(gt_step2)
        
        stacknum = 1
        X2_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
        s1_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
        for t in range(stacknum):
            #iterate through the batch
            for i in range(X.shape[0]):
                if(args.dataset=='blender'or args.dataset=='defocusnet' or args.dataset=='nyu'):
                    fd=sample_batch['fdist'][i].item()
                    f=sample_batch['f'][i].item()
                    k=sample_batch['kcam'][i].item()
                    if(not args.aif):
                        X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*k*(fd-f)
                elif(args.dataset=='ddff'):
                    fd=foc_dist[i].item()
                    if(not args.aif):
                        X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*kcam*(fd-f)
                # if(not aif):
                #     s1_fcs[i, t:(t + 1), :, :] = s1_fcs[i, t:(t + 1), :, :]*(focus_distance)/3.0
        X2_fcs = X2_fcs.float().to(device_comp)
        s1_fcs = s1_fcs.float().to(device_comp)
        
        pred_depth,pred_blur,corrected_blur=model(X,camind=args.camind,camparam=X2_fcs)
        
        #gt_blur=torch.cat((gt_blur,torch.flatten(gt_step1.detach().cpu())))
        #pred_blur=torch.cat((pred_blur,torch.flatten(output_step1.detach().cpu())))

        minblur_=torch.min(pred_blur).item()
        if(minblur_<minblur):   
            minblur=minblur_
        maxblur_=torch.max(pred_depth).item()
        if(maxblur_>maxblur):
            maxblur=maxblur_

        #output_step1=output_step1*(0.1-2.9e-3)*7 
        blurpred=pred_blur
        #calculate s2 provided that s2>s1
        s2est=0.1*1./(1-blurpred)
        #blur mse
        if(args.dataset=='blender' or args.dataset=='defocusnet'):
            blurmse=torch.mean(torch.square(pred_blur*args.blurclip-blur)[mask>0]).item()
            meanblurmse+=blurmse
        #calculate MSE value
        denorm_depth=denormalize(pred_depth)
        if(args.out_depth):
            mse=torch.mean(torch.square(denorm_depth-depth)[mask>0]).item()
        else:
            mse=torch.mean(torch.square(focus_distance/denorm_depth-focus_distance/depth)[mask>0]).item()
            mse2=torch.mean(torch.square(denorm_depth-depth)[mask>0]).item()
        meanMSE+=mse
        meanMSE2+=mse2

        if(calc_distmse):
            squareder=torch.square(depth*args.depthscale-gt_step2)
            gtround=torch.round(gt_step2*10,decimals=0)
            for i in range(1,len(distmse)+1):
                selected_val=squareder[gtround==i]
                selected_blur=corrected_blur[gtround==i]
                #mask_sum=torch.sum(mask[gtround==i]).item()
                er=(torch.mean(selected_val)).item()
                b=(torch.mean(selected_blur)).item()
                if(not(math.isnan(er) or math.isnan(b))):
                    distmse[i-1]+=er
                    distblur[i-1]+=b
                    distsum[i-1]+=1

        blur=torch.sum(pred_blur*mask).item()/torch.sum(mask).item()
        meanblur+=blur
        if(args.dataset=='blender' or args.dataset=='defocusnet'):
            gtblur=torch.sum(blur*mask).item()/torch.sum(mask).item()
            gt_meanblur+=gtblur
    if(calc_distmse):
        print('\ndistance wise error (distances rounded to the shown value): ')
        mse_=distmse/distsum
        blur_=distblur/distsum
        mse_=mse_[~torch.isnan(mse_)]
        blur_=blur_[~torch.isnan(blur_)]
        values=np.arange(0.1,(len(mse_)+1)*0.1,0.1)
        print('distances:')
        for i,v in enumerate(values):
            print("%4.3f"%(v),end=",")
        print('\nMSE:')
        for i,v in enumerate(values):
            print("%4.3f"%(mse_[i].item()),end=",")
        print('')
    return meanMSE/len(loader), meanMSE2/len(loader),meanblurmse/len(loader),meanblur/len(loader),gt_meanblur/len(loader),minblur,maxblur

def train_model(loader):
    criterion = torch.nn.MSELoss()
    #criterion=F.smooth_l1_loss(reduction='none')
    optimizer = optim.Adam(model_params, lr=args.lr)

    ##### Training
    print("Total number of epochs:", args.epochs)
    for e_iter in range(args.epochs):
        epoch_iter = e_iter
        loss_sum, iter_count,absloss_sum= 0,0,0
        depthloss_sum,blurloss_sum=0,0
        blur_sum=0
        mean_blur=0
        for st_iter, sample_batch in enumerate(loader):
            if(args.dataset=="nyu"):
                X=sample_batch['rgb'].float().to(device_comp)
                depth=sample_batch['depth'].float().to(device_comp)
                blur=sample_batch['blur'].float().to(device_comp)
                depth=torch.unsqueeze(depth,dim=1)
                blur=torch.unsqueeze(blur,dim=1)
                stacknum=X.shape

                # plot data
                # import matplotlib.pyplot as plt
                # img=X[10,:,:,:].detach().cpu().permute(1,2,0)
                # plt.imshow(img)
                # plt.show()
                # d=gt_step2[10,0,:,:].detach().cpu()
                # plt.imshow(d)
                # plt.show()

                # f, axarr = plt.subplots(1,2)
                # axarr[0].imshow(img)
                # axarr[1].imshow(d)
                # plt.show()

            else:
                # Setting up input and output data
                X = sample_batch['input'][:,0,:,:,:].float().to(device_comp)
                depth=sample_batch['depth'].float().to(device_comp)
                blur=sample_batch['blur'].float().to(device_comp)
            focus_distance=sample_batch['fdist']
            focus_distance=torch.unsqueeze(focus_distance,dim=2).unsqueeze(dim=3)
            focus_distance=torch.repeat_interleave(focus_distance,depth.shape[2],dim=2).repeat_interleave(depth.shape[3],dim=3)
            focus_distance=focus_distance.to(device_comp)
                
            optimizer.zero_grad()
            
            mask=(focus_distance/depth>args.s2limits[0])*(focus_distance/depth<args.s2limits[1]).int()

            # we only use focal stacks with a single image
            stacknum = 1

            '''
            convert blur_pix that is being estimated by the model
            into
            |s2-s1|/s1  by multiplying blur_pix_est by (s1-f)/kcam
            which is camera independent because it only depends on the distances; not on camera parameters.
            Also see dofNet_arch3.py comments on the calculations
            '''
            X2_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
            s1_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
            for t in range(stacknum):
                #iterate through the batch
                for i in range(X.shape[0]):
                    focus_distance=sample_batch['fdist'][i].item()
                    f=sample_batch['f'][i].item()
                    k=sample_batch['kcam'][i].item()     
                    if(not args.aif):
                        X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]*k*(focus_distance-f)
                        # s1_fcs[i, t:(t + 1), :, :] = s1_fcs[i, t:(t + 1), :, :]*(focus_distance)/3.0

            X2_fcs = X2_fcs.float().to(device_comp)
            s1_fcs = s1_fcs.float().to(device_comp)

            # Forward and compute loss
            pred_depth,pred_blur,_=model(X,camind=args.camind,camparam=X2_fcs)

            blur_sum+=torch.sum(pred_blur[mask>0]).item()/torch.sum(mask)
            norm_depth=normalize(depth)
            depth_loss=criterion(pred_depth[mask>0], (norm_depth)[mask>0])
            #we don't train blur if input images are AiF
            if(args.aif):
                blur_loss=0 
            else:
                blur_loss=criterion(pred_blur[mask>0], (blur/args.blurclip)[mask>0])
            loss=depth_loss+blur_loss*args.blurweight

            absloss=torch.sum(torch.abs(pred_blur-blur)[mask>0])/torch.sum(mask)
            absloss_sum+=absloss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            # gradient=0
            # n=0
            # for p in optimizer.param_groups[0]['params']:
            #     if(p.grad is not None):
            #         gradient+=torch.mean(p.grad).item()
            #         n+=1
            # print('mean gradient:'+str(gradient/n))
            optimizer.step()

            # Training log
            loss_sum += loss.item()
            iter_count += 1.
            if(args.aif):
                blurloss_sum+=0
            else:
                blurloss_sum+=blur_loss.item()
            depthloss_sum+=depth_loss.item()

            if (st_iter + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch_iter + 1, args.epochs, st_iter + 1, total_steps, loss_sum / iter_count))
    
                absloss_sum=0
                depth_sum,blur_sum=0,0
                depthloss_sum,blurloss_sum=0,0

                total_iter = total_steps * epoch_iter + st_iter
                loss_sum, iter_count = 0,0

        # Save model
        if (epoch_iter+1) % 10 == 0:
            print('saving model')
            torch.save(model.state_dict(), args.savepath +expname+ '\\model.pth')
            depthMSE,valueMSE,blurloss,meanblur,gtmeanblur,minblur,maxblur=eval(loaders[1])
            print('depth MSE: '+str(depthMSE))
            print('s1/depth MSE: '+str(valueMSE))
            print('blur loss = '+str(blurloss))
            print('mean blur = '+str(meanblur))

def main():
    train_model(loaders[0])

if __name__ == "__main__":
    main()

#datapath='C:\\Users\\lahir\\focalstacks\\datasets\\mediumN1\\'
#focalblender.get_data_stats(datapath,50)
'''
fdist of DDFF 
tensor([[0.2800, 0.2511, 0.2222, 0.1933, 0.1644, 0.1356, 0.1067, 0.0778, 0.0489,
         0.0200]])
'''

'''
#plotting distribution of blur
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
f=3e-3
p=3.1e-3/256
s=1
N=1
for f in [3e-3,4e-3,5e-3]:
    s1 = np.random.uniform(0.1,1.5,1000)
    s2 = np.random.uniform(0.0,2.0,1000)
    s2=np.random.normal(loc=1.0,scale=0.1,size=1000)
    blur=np.abs(s1-s2)/s2*1/(s1-f) * f**2/N *1/p * s
    density = stats.gaussian_kde(blur)
    bins = np.linspace(0.1, 2.0, 1000)
    n,bins = np.histogram(np.array(blur), bins)
    plt.plot(bins, density(bins),label='f=%1.0fmm'%(f*1000))
ax = plt.gca()
# Hide X and Y axes label marks
ax.yaxis.set_tick_params(labelleft=False)
# Hide Y axes tick marks
ax.set_yticks([])
plt.legend()
plt.xlabel('Blur in pixles')
plt.ylabel('Density')
plt.savefig('blur_distF.png', dpi=500)
plt.show()
'''