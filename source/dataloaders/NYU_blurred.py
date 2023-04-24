import torch
import torch.utils.data
from torchvision import transforms

from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as F

# N=1
# f =50.0e-3
# s1= 3.0
# px =36e-6

# k=f**2/N/px
# k=1/k
# k

CROP_PIX=15

#read kcams.txt file
def read_kcamfile(file):
    d = {}
    with open(file) as f:
        for line in f:
            if(len(line)<2):
                continue
            (key, val) = line.split()
            try:
                d[key] = float(val)
            except:
                 d[key] = val
    return d
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
    """Focal place dataset."""

    def __init__(self, rgbpath,depthpath, transform_fnc=None,blur=1,aif=0,fstack=0, max_dpt=1.,blurclip=1.,kcampath=None,
                 out_depth=False):

        self.rgbpath=rgbpath
        self.depthpath=depthpath

        self.transform_fnc = transform_fnc

        self.blur=blur
        self.aif=aif
        self.fstack=fstack
        self.blurclip=blurclip
        self.kcampath=kcampath
        self.out_depth=out_depth
        if(kcampath):
            self.kcamdict=read_kcamfile(kcampath)
            #calculate kcam based on parameters in file
            N=self.kcamdict['N']
            self.f=self.kcamdict['f']
            px=self.kcamdict['px']
            self.s1=self.kcamdict['focus']
            self.kcam=1/(self.f**2/N/px)
            print('kcam:'+str(self.kcam))
            print('f:'+str(self.f))

            print(self.kcam*(3-self.f))

        ##### Load and sort all images
        self.imglist_rgb = [f for f in listdir(rgbpath) if isfile(join(rgbpath, f)) and f[-4:] == ".png"]
        self.imglist_dpt = [f for f in listdir(depthpath) if isfile(join(depthpath, f)) and f[-4:] == ".png"]

        print("Total number of samples", len(self.imglist_dpt), "  Total number of seqs", len(self.imglist_dpt))

        self.imglist_rgb.sort()
        self.imglist_dpt.sort()

        self.max_dpt = max_dpt

    def __len__(self):
        return int(len(self.imglist_dpt))

    def __getitem__(self, idx):
        fdist=np.zeros((0))

        ##### Read and process an image
        #read depth image
        img_dpt = cv2.imread(self.depthpath + self.imglist_dpt[idx],cv2.IMREAD_UNCHANGED)
        #convert from mm to m
        img_dpt=img_dpt/1000.
        #img_dpt_scaled = np.clip(img_dpt, 0., 1.9)
        #mat_dpt_scaled = img_dpt_scaled / 1.9
        mat_dpt_scaled = img_dpt/self.max_dpt
        mat_dpt = mat_dpt_scaled.copy()[:, :, np.newaxis]
        if(not self.out_depth):
            mat_dpt = mat_dpt/self.s1

        #read rgb image
        im = cv2.imread(self.rgbpath + self.imglist_rgb[idx],cv2.IMREAD_UNCHANGED)
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
        # _,w,h=image.shape
        # cropped=F.crop(image,CROP_PIX,CROP_PIX,w-CROP_PIX,h-CROP_PIX)
        return image
    


def load_data(rgbpath,depthpath, blur,train_split,fstack,
              WORKERS_NUM, BATCH_SIZE, MAX_DPT=1.,blurclip=1.,kcampath=None,out_depth=False):
    tr=transforms.Compose([
        Transform(),
        transforms.RandomCrop((256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
        ])
    img_dataset = ImageDataset(rgbpath=rgbpath,depthpath=depthpath,blur=blur,transform_fnc=tr,
                               fstack=fstack, max_dpt=MAX_DPT,
                               blurclip=blurclip,kcampath=kcampath,out_depth=out_depth)

    indices = list(range(len(img_dataset)))
    split = int(len(img_dataset) * train_split)

    indices_train = indices[:split]
    indices_valid = indices[split:]

    dataset_train = torch.utils.data.Subset(img_dataset, indices_train)
    dataset_valid = torch.utils.data.Subset(img_dataset, indices_valid)

    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=WORKERS_NUM, batch_size=BATCH_SIZE, shuffle=True)
    loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid, num_workers=1, batch_size=1, shuffle=False)

    total_steps = int(len(dataset_train) / BATCH_SIZE)
    print("Total number of steps per epoch:", total_steps)
    print("Total number of training sample:", len(dataset_train))
    print("Total number of validataion sample:", len(dataset_valid))

    return [loader_train, loader_valid], total_steps


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
X mean=0.39910310504313146
depth min=0.7129999995231628
depth max=9.98900032043457
depth mean=2.9440300959878383
blur min=0.0
blur max=33.832828521728516
blur mean=6.4919104153118425
'''


def get_data_stats(blurclip):
    depthpath='C:\\Users\\lahir\\data\\nyu_depth\\noborders\\depth\\'
    rgbpath='C:\\Users\\lahir\\data\\nyu_depth\\noborders\\refocused1\\'
    kcampath='C:\\Users\\lahir\\data\\nyu_depth\\noborders\\refocused1\\camparam.txt'
    loaders, total_steps = load_data(rgbpath=rgbpath,depthpath=depthpath,blur=1,train_split=0.8,fstack=0,WORKERS_NUM=0,
        BATCH_SIZE=1,MAX_DPT=1.0,blurclip=1,kcampath=kcampath,out_depth=True)
    print('stats of train data')
    depthlist=get_loader_stats(loaders[0])
    #plot histogram of GT depths
    depthlist=depthlist.numpy()
    _ = plt.hist(depthlist, bins='auto') 
    plt.title('Depth stats of NYU dataset. min='+str(min(depthlist))+' max='+str(max(depthlist))+' mean='+str(np.mean(depthlist)))
    plt.show()
    print('______')

#data statistics of the input images
def get_loader_stats(loader):
    print('getting NUY stats...')
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

# get_data_stats(1)

'''
blur_thres=7.0
p=3.1/256*1e-3 # pixel width in m
N=2
f=6e-3
s2range=[0.1,2.0]
s1range=[0.1,1.5]

get_workable_s1s2ranges(p,N,f,s2range,s1range,blur_thres)
'''







