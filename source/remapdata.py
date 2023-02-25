from os import listdir, mkdir,rename
from os.path import isfile, join, isdir
import numpy as np


path='C://Users//lahir//focalstacks//datasets//mediumN1-10_test_remapped//'
depthfiles=[f for f in listdir(path) if isfile(join(path, f)) and f[-7:] == "Dpt.exr"]

kcams=[float(file.split('_')[1]) for file in depthfiles]
unique_kcams,counts=np.unique(kcams,return_counts=True)
unique_kcams=np.sort(unique_kcams)

for file in depthfiles:
    splt=file.split('_')
    kcam=float(splt[1])
    arg=np.argwhere(unique_kcams==kcam)[0][0]
    newname=splt[0]+'_'+str(arg)+'_'+('_'.join(splt[2:]))
    rename(path+file,path+newname)

#read kcams.txt file
d = {}
with open(path+"kcams.txt") as f:
    for line in f:
       (key, val) = line.split()
       d[int(key)] = float(val)



