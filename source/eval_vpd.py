import os
import cv2
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys 
#include stable diffision path
sys.path.append('C:\\Users\\lahir\\code\\defocusVdepth\\stable-diffusion\\')
from arch.model import VPDDepth
import utils_depth.metrics as metrics
import logging
logger=logging
import utils
from configs.nyu_options import NYUOptions

# from models_depth.optimizer import build_optimizers

import test
import importlib
import time
from os.path import join


opt = NYUOptions()
args = opt.initialize().parse_args()

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']

device_id=0

args.shift_window_test=True
args.flip_test=True
print(args)


model = VPDDepth(args=args).to(device_id)
model_params = model.parameters()

#get model and load weights
if args.trained_model:
    from collections import OrderedDict
    print('loading weigths to the model....')
    logging.info('loading weigths to the model....')
    cudnn.benchmark = True
    #load weights to the model
    print('loading from :'+str(args.trained_model))
    logging.info('loading from :'+str(args.trained_model))
    model_weight = torch.load(args.trained_model)['model']
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight, strict=False)
    print('loaded weights')
    logging.info('loaded weights')  

#create the name of the model
pretrain = args.pretrained.split('.')[0]
maxlrstr = str(args.max_lr).replace('.', '')
minlrstr = str(args.min_lr).replace('.', '')
layer_decaystr = str(args.layer_decay).replace('.', '')
weight_decaystr = str(args.weight_decay).replace('.', '')
num_filter = str(args.num_filters[0]) if args.num_deconv > 0 else ''
num_kernel = str(args.deconv_kernels[0]) if args.num_deconv > 0 else ''
name = [args.dataset, str(args.batch_size), pretrain.split('/')[-1], 'deconv'+str(args.num_deconv), \
        str(num_filter), str(num_kernel), str(args.crop_h), str(args.crop_w), maxlrstr, minlrstr, \
        layer_decaystr, weight_decaystr, str(args.epochs)]
if args.exp_name != '':
        name.append(args.exp_name)
exp_name = '_'.join(name)
print('This experiments: ', exp_name)

# Dataset setting
dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path,'rgb_dir':args.rgb_dir, 'depth_dir':args.depth_dir,'is_blur':args.is_blur}
dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

val_dataset = get_dataset(**dataset_kwargs, is_train=False)

sampler_val = torch.utils.data.DistributedSampler(
        val_dataset, num_replicas=1, rank=0, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                         num_workers=0,pin_memory=True)


if __name__ == "__main__":
    test.vali_dist(val_loader,model,device_id,args,logger,args.geometry_model)