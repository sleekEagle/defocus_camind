import torch
import torch.nn as nn   
import torch.optim as optim
import torch.utils.data
import numpy as np
import math
import argparse
from dataloaders import NYU_blurred, focalblender
from arch import dofNet_arch1,dofNet_arch4
import sys
import os
import util_func
from torch.optim.lr_scheduler import StepLR
from pathlib import Path

ch_inp_num = 192
ch_out_num = 1
model = dofNet_arch1.AENet(ch_inp_num, 1, 16, flag_step2=True)
model_params = model.parameters()

img=torch.rand((1,192,480,480))

model(img,inp=1,k=192)