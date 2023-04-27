import h5py
import numpy as np
import matplotlib.pyplot as plt
import imageio
import scipy 


splits="C:\\Users\\lahir\\data\\matlabfiles\\splits.mat"
mat=scipy.io.loadmat(splits)
test_idx=mat['testNdxs']
train_idx=mat['trainNdxs']


outpath='C:\\Users\\lahir\\data\\nyu_depthnew\\'
with h5py.File('C:\\Users\\lahir\\data\\matlabfiles\\nyu_depth_v2_labeled.mat', 'r') as f:
    depths=f['depths'][:,:,:]
    imgs=f['images'][:,:,:,:]

for i in range(depths.shape[0]):
    #write depth image
    d=depths[i,:,:].transpose()
    d=d*1000.
    d=d.astype(np.uint16)
    rgb=imgs[i,:,:]
    imageio.imwrite(outpath+'depth\\'+str(i+1)+'.png', d)
    imageio.imwrite(outpath+'rgb\\'+str(i+1)+'.png', rgb.transpose())
    #write rgb image
    # rgb=imgs[i,:,:,:]
    # break


#remap files
from os.path import isfile, join, isdir
from os import listdir, mkdir
import os

input_path='C:\\Users\\lahir\\data\\nuy_depth\\refocused8\\'
output_path='C:\\Users\\lahir\\data\\nuy_depth\\refocused8_r\\'
files = [f for f in listdir(input_path) if isfile(join(input_path, f)) and f[-4:] == ".png"]

for file in files:
    newname=str(int(file.split('_')[0]))+'.png'
    os.popen('copy '+input_path+file+' '+output_path+newname)


'''
crop the border or RGB and depth images
'''
import cv2
CROPPIX=20
rgbpath='C:\\Users\\lahir\\data\\nyu_depth\\rgb\\'
depthpath='C:\\Users\\lahir\\data\\nyu_depth\\depth\\'
rgbfiles = [f for f in listdir(rgbpath) if isfile(join(rgbpath, f)) and f[-4:] == ".png"]
depthfiles = [f for f in listdir(depthpath) if isfile(join(depthpath, f)) and f[-4:] == ".png"]
rgbfiles.sort()
depthfiles.sort()

outpath='C:\\Users\\lahir\\data\\nyu_depth\\noborders\\'

for i,rgb in enumerate(rgbfiles):
    rgbimg = cv2.imread(rgbpath+rgb,cv2.IMREAD_UNCHANGED)
    depthimg = cv2.imread(depthpath+depthfiles[i],cv2.IMREAD_UNCHANGED)
    #crop the boarder
    h,w,_=rgbimg.shape
    rgbcrop=rgbimg[CROPPIX:h-CROPPIX,CROPPIX:w-CROPPIX,:]
    depthcrop=depthimg[CROPPIX:h-CROPPIX,CROPPIX:w-CROPPIX]
    #write to file
    cv2.imwrite(outpath+"rgb\\"+rgb,rgbcrop)
    cv2.imwrite(outpath+"depth\\"+depthfiles[i],depthcrop)






