#load the trained DFV model that is used to predict depth (s2)
import argparse
import cv2
from DFV.models import DFFNet as DFFNet
import os
import time
from DFV.models.submodule import *

from torch.utils.data import DataLoader
from DFV.dataloader import FoD500Loader

import  matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='defocu_camind')
parser.add_argument('--loadmodel', default='C://Users//lahir//code//defocus//models//DFV//mediumN1.tar', help='DFV model path')
parser.add_argument('--data_path', default='C://Users//lahir//focalstacks//datasets//mediumN1-3//', help='data path to focal stacks')
args = parser.parse_args()

# construct DFV model and load weights
model = DFFNet( clean=False,level=4, use_diff=1)
model = nn.DataParallel(model)
model.cuda()
ckpt_name = os.path.basename(os.path.dirname(args.loadmodel))# we use the dirname to indicate training setting

if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    print('run with random init')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


dataset_validation, _ = FoD500Loader(args.data_path, n_stack=6)
dataloader = torch.utils.data.DataLoader(dataset=dataset_validation, num_workers=1, batch_size=1, shuffle=False)

for inx, (img_stack, gt_disp, foc_dist) in enumerate(dataloader):
    break







