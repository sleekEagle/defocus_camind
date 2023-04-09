import h5py
import numpy as np
import matplotlib.pyplot as plt
import imageio

outpath='C:\\Users\\lahir\\data\\nuy_depth\\'
with h5py.File('C:\\Users\\lahir\\data\\nyu_depth_v2_labeled.mat', 'r') as f:
    depths=f['depths'][:,:,:]
    imgs=f['images'][:,:,:,:]

for i in range(depths.shape[0]):
    #write depth image
    d=depths[i,:,:].transpose()
    d=d*1000.
    d=d.astype(np.uint16)
    imageio.imwrite(outpath+'depth\\'+str(i+1)+'.png', d)
    #write rgb image
    # rgb=imgs[i,:,:,:]
    # break


#remap files
from os.path import isfile, join, isdir
from os import listdir, mkdir
import os

input_path='C:\\Users\\lahir\\data\\nuy_depth\\refocused4\\'
output_path='C:\\Users\\lahir\\data\\nuy_depth\\refocused4_r\\'
files = [f for f in listdir(input_path) if isfile(join(input_path, f)) and f[-4:] == ".png"]

for file in files:
    newname=str(int(file.split('_')[0]))+'.png'
    os.popen('copy '+input_path+file+' '+output_path+newname)
