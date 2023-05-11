import torch
import torch.utils.data
from torchvision import transforms

from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as F
import scipy 
from pathlib import Path

# N=1.0
# f =16e-3
# s1= 1
# px =36e-6

# k=f**2/N/px
# k=1/k
# k
# mind=0.71
# maxd=9.99
# s2=np.arange(mind,maxd,0.01)

# blur=get_blur(s1,s2,f,k)
# min(blur)
# max(blur)

# plt.plot(s2,blur)
# plt.show()

'''
Hard code the camera parameters
'''

'''
# to calculate circle of confusion
output |s2-s1|/s2
'''
def get_blur(s1,s2,f,kcam):
    blur=abs(s2-s1)/s2 * 1/(s1-f)*1/kcam
    return blur

'''
All in-focus image is attached to the input matrix after the RGB image

focus_dist - available focal dists in the dataset
req_f_indx - a list of focal dists we require.

fstack=0
a single image from the focal stack will be returned.
The image will be randomly selected from indices in req_f_indx
fstack=1
several images comprising of the focal stack will be returned. 
indices of the focal distances will be selected from req_f_indx

if aif=1 : 
    all in focus image will be appended at the begining of the focal stack

input matrix channles :
[batch,image,rgb_channel,256,256]

if aif=1 and fstack=1
image 0 : all in focus image
image 1:-1 : focal stack
if aif=0
image 0:-1 : focal stack

if fstack=0

output: [depth, blur1, blur2,]
blur1,blur2... corresponds to the focal stack
'''

class ImageDataset(torch.utils.data.Dataset):
    # datapath="C:\\Users\\lahir\\data\\nyu_depth\\noborders"
    # datanum=1
    # f='267.png'
    # int(f[:-4])
    # imgp=rgbpath/"1261.png"
    # cv2.imread(str(imgp),cv2.IMREAD_UNCHANGED)

    def __init__(self,rgb_list,dpt_list,transform_fnc=None,blur=1,aif=0, max_dpt=1.,blurclip=1.,
                 out_depth=False,f=3e-3,s1=2.0,kcam=1.4):

        self.transform_fnc = transform_fnc
        self.blur=blur
        self.aif=aif
        self.blurclip=blurclip
        self.out_depth=out_depth
        self.max_dpt = max_dpt
        self.rgb_list=rgb_list
        self.dpt_list=dpt_list
        self.s1=s1
        self.f=f
        self.kcam=kcam


    def __len__(self):
        return int(len(self.rgb_list))

    def __getitem__(self, idx):
        fdist=np.zeros((0))

        ##### Read and process an image
        #read depth image
        img_dpt = cv2.imread(str(self.dpt_list[idx]),cv2.IMREAD_UNCHANGED)
        #convert from mm to m
        img_dpt=img_dpt/1000.
        mat_dpt_scaled = img_dpt/self.max_dpt
        mat_dpt = mat_dpt_scaled.copy()[:, :, np.newaxis]
        if(self.out_depth==0):
            mat_dpt = mat_dpt/self.s1

        #read rgb image
        im = cv2.imread(str(self.rgb_list[idx]),cv2.IMREAD_UNCHANGED)
        img_rgb = np.array(im)
        mat_rgb = img_rgb.copy() / 255.
            
        img_blur = get_blur(self.s1, img_dpt,self.f,self.kcam)
        img_blur = img_blur / self.blurclip
        #img_msk = np.clip(img_msk, 0, 1.0e-4) / 1.0e-4
        mat_blur = img_blur.copy()[:, :, np.newaxis]

        data=np.concatenate((mat_rgb,mat_dpt,mat_blur),axis=2)
        data=torch.from_numpy(data)

        fdist=np.concatenate((fdist,[self.s1]),axis=0)
                
        if self.transform_fnc:
            data_tr = self.transform_fnc(data)
        sample = {'rgb': data_tr[:3,:,:], 'depth': data_tr[3,:,:],'blur':data_tr[4,:,:],'fdist':fdist,'kcam':self.kcam,'f':self.f}
        return sample


class Transform(object):
    def __call__(self, image):
        image=torch.permute(image,(2,0,1))
        return image

datapath='C:\\Users\\lahir\\data\\DSLR\\dfd_indoor\\dfd_dataset_indoor_N2_8\\'
#how many images are used to train the model
train_n=10

def load_data(datapath,train_n,blur,
              WORKERS_NUM, BATCH_SIZE, MAX_DPT=1.,blurclip=1.,out_depth=False):
    
    #reading splits
    p=Path(datapath)
    train_depth=p/"depth"/"train"
    train_rgb=p/"rgb"/"train"
    test_depth=p/"depth"/"test"
    test_rgb=p/"rgb"/"test"

    #get all files
    rgb1 = [train_rgb/f for f in listdir(train_rgb) if (isfile(join(train_rgb, f)) and f[-4:] == ".JPG")]
    rgb2 = [test_rgb/f for f in listdir(test_rgb) if (isfile(join(test_rgb, f)) and f[-4:] == ".JPG")]
    dpt1 = [train_depth/f for f in listdir(train_depth) if (isfile(join(train_depth, f)) and f[-4:] == ".png")]
    dpt2 = [test_depth/f for f in listdir(test_depth) if (isfile(join(test_depth, f)) and f[-4:] == ".png")]
    rgb1.sort()
    rgb2.sort()
    dpt1.sort()
    dpt2.sort()

    rgb_train=rgb1[:train_n]
    dpt_train=dpt1[:train_n]
    rgb_test=rgb1[train_n:]+rgb2
    dpt_test=dpt1[train_n:]+dpt2
    
    tr_train=transforms.Compose([
        Transform(),
        transforms.RandomCrop((256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
        ])
    tr_test=transforms.Compose([
        Transform(),
        transforms.RandomCrop((256,256)),
        ])
    train_dataset = ImageDataset(rgb_train,dpt_train,blur=blur,transform_fnc=tr_train,
                               max_dpt=MAX_DPT,
                               blurclip=blurclip,out_depth=out_depth)
    
    test_dataset = ImageDataset(rgb_test,dpt_test,blur=blur,transform_fnc=tr_train,
                               max_dpt=MAX_DPT,
                               blurclip=blurclip,out_depth=out_depth)

    loader_train = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=WORKERS_NUM, batch_size=BATCH_SIZE, shuffle=True)
    loader_test = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=WORKERS_NUM, batch_size=BATCH_SIZE, shuffle=True)

    # total_steps = int(len(train_dataset) / BATCH_SIZE)
    # print("Total number of steps per epoch:", total_steps)
    # print("Total number of training sample:", len(train_dataset))
    # print("Total number of validataion sample:", len(test_dataset))

    return [loader_train,loader_test]


# depthpath='C:\\Users\\lahir\\data\\nyu_depth\\noborders\\'
# # rgbpath='C:\\Users\\lahir\\data\\nuy_depth\\refocused5\\'
# # kcampath='C:\\Users\\lahir\\data\\nuy_depth\\refocused5\\camparam.txt'
# rgbpath=depthpath+"refocused1\\"
# depthpath=depthpath+"depth\\"
# kcampath=depthpath+"refocused1\\camparam.txt"
# loaders, total_steps=load_data(rgbpath=rgbpath,depthpath=depthpath,blur=1,train_split=0.8,fstack=0,WORKERS_NUM=0,
#         BATCH_SIZE=1,MAX_DPT=1,blurclip=1,kcampath=kcampath)

# blurclip=1
# loaders, total_steps = load_data(rgbpath=rgbpath,depthpath=depthpath,blur=1,train_split=0.8,fstack=0,WORKERS_NUM=0,
#         BATCH_SIZE=1,MAX_DPT=1.0,blurclip=blurclip,kcampath=kcampath)
# for st_iter, sample_batch in enumerate(loaders[0]):
#     rgb=sample_batch['rgb']
#     depth=sample_batch['depth']
#     blur=sample_batch['blur']
#     break

# import matplotlib.pyplot as plt
# img=rgb[0,:,:,:]
# img=torch.permute(img,(1,2,0))

# d=torch.permute(depth,(1,2,0))
# b=torch.permute(blur,(1,2,0))

# d[203,100,0]
# b[203,100,0]
# abs(2.0730-2.0)/2.0730*1/(2.0-50e-3)*1/0.027359999999999992

# f, axarr = plt.subplots(1,2)
# axarr[0].imshow(d)
# axarr[1].imshow(b)
# plt.show()

# plt.imshow(b)
# plt.savefig('C:\\Users\\lahir\\data\\nuy_depth\\blur.png')

'''
X min=0.0
X max=1.0
X mean=0.49368787379046664
depth min=0.24500000476837158
depth max=8.345999717712402
depth mean=2.831630492210388
blur min=0.0
blur max=2.562152147293091
blur mean=0.17454538345336915
'''


def get_data_stats():
    loaders=load_data(datapath=datapath,train_n=train_n,blur=1,WORKERS_NUM=0,
            BATCH_SIZE=1,out_depth=True)
    print('stats of train data')
    depthlist=get_loader_stats(loaders[1])
    #plot histogram of GT depths
    depthlist=depthlist.numpy()
    _ = plt.hist(depthlist, bins='auto') 
    plt.title('Depth stats of DSLR dataset. min='+str(min(depthlist))+' max='+str(max(depthlist))+' mean='+str(np.mean(depthlist)))
    plt.show()
    print('______')

#data statistics of the input images
def get_loader_stats(loader):
    print('getting DSLR stats...')
    xmin,xmax,xmean,count=100,0,0,0
    depthmin,depthmax,depthmean=100,0,0
    blurmin,blurmax,blurmean=100,0,0
    depthlist=torch.empty(0)
    for st_iter, sample_batch in enumerate(loader):
        # Setting up input and output data
        X=sample_batch['rgb']
        depth=sample_batch['depth']
        blur=sample_batch['blur']
        gt_step1=blur.float()
        gt_step2=depth.float()
        gt_step1=torch.unsqueeze(gt_step1,dim=1)
        gt_step2=torch.unsqueeze(gt_step2,dim=1)

        xmin_=torch.min(X).cpu().item()
        if(xmin_<xmin):
            xmin=xmin_
        xmax_=torch.max(X).cpu().item()
        if(xmax_>xmax):
            xmax=xmax_
        xmean+=torch.mean(X).cpu().item()
        count+=1
        t=torch.flatten(gt_step2)
        depthlist=torch.concat((depthlist,t),axis=0)
        depthmin_=torch.min(gt_step2).cpu().item()
        if(depthmin_<depthmin):
            depthmin=depthmin_
        depthmax_=torch.max(gt_step2).cpu().item()
        if(depthmax_>depthmax):
            depthmax=depthmax_
        depthmean+=torch.mean(gt_step2).cpu().item()

        blurmin_=torch.min(gt_step1).cpu().item()
        if(blurmin_<blurmin):
            blurmin=blurmin_
        blurmax_=torch.max(gt_step1).cpu().item()
        if(blurmax_>blurmax):
            blurmax=blurmax_
        blurmean+=torch.mean(gt_step1).cpu().item()

    print('X min='+str(xmin))
    print('X max='+str(xmax))
    print('X mean='+str(xmean/count))

    print('depth min='+str(depthmin))
    print('depth max='+str(depthmax))
    print('depth mean='+str(depthmean/count))

    print('blur min='+str(blurmin))
    print('blur max='+str(blurmax))
    print('blur mean='+str(blurmean/count))
    return depthlist

# get_data_stats()

'''
blur_thres=7.0
p=3.1/256*1e-3 # pixel width in m
N=2
f=6e-3
s2range=[0.1,2.0]
s1range=[0.1,1.5]

get_workable_s1s2ranges(p,N,f,s2range,s1range,blur_thres)
'''







