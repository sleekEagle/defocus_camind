import argparse
import cv2
from models import DFFNet as DFFNet
import os
import time
from models.submodule import *

from torch.utils.data import DataLoader


import  matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from source import util_func

'''
Main code for Ours-FV and Ours-DFV test on FoD500 dataset  
'''

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='DFVDFF')
parser.add_argument('--data_path', default='C://Users//lahir//focalstacks//datasets//mediumN1//',help='test data path')
parser.add_argument('--loadmodel', default='C://Users//lahir//code//defocus//models//DFV//best.tar', help='model path')
parser.add_argument('--outdir', default='./FoD500/',help='output dir')
parser.add_argument('--use_diff', default=1, choices=[0,1], help='if use differential images as input, change it according to the loaded checkpoint!')
parser.add_argument('--level', type=int, default=4, help='num of layers in network, please take a number in [1, 4]')
args = parser.parse_args()

# !!! Only for users who download our pre-trained checkpoint, comment the next four line if you are not !!!
'''
if os.path.basename(args.loadmodel) == 'DFF-DFV.tar' :
    args.use_diff = 1
else:
    args.use_diff = 0
'''
# construct model
model = DFFNet( clean=False,level=args.level, use_diff=args.use_diff)
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


def disp2depth(disp):
    dpth = 1 / disp
    dpth[disp == 0] = 0
    return dpth


def main(image_size = (256, 256)):
    model.eval()

    loaders, total_steps = util_func.load_data(args.data_path,blur=0,aif=0,train_split=0.8,fstack=1,WORKERS_NUM=0,
    BATCH_SIZE=1,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,100000],REQ_F_IDX=[0,1,2,3,4],MAX_DPT=1.)
    TrainImgLoader,ValImgLoader=loaders[0],loaders[1]

    # metric prepare
    test_num = len(ValImgLoader)
    time_list = []
    std_sum = 0

    for inx, sample_batch in enumerate(ValImgLoader):
        img_stack=sample_batch['input'].float()
        gt_disp=sample_batch['output'][:,0,:,:]
        gt_disp=torch.unsqueeze(gt_disp,dim=1).float()
        foc_dist=sample_batch['fdist'].float()

        # if inx not in  [5, 64,67]:continue
        if inx % 10 == 0:
            print('processing: {}/{}'.format(inx, test_num))

        img_stack = Variable(torch.FloatTensor(img_stack)).cuda()

        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            pred_disp, std, focusMap = model(img_stack, (foc_dist))
            torch.cuda.synchronize()
            ttime = (time.time() - start_time); print('time = %.2f' % (ttime*1000) )
            std_sum += std.mean()

        pred_disp = pred_disp.squeeze().cpu().numpy()[:image_size[0], :image_size[1]]

        pred_dpth = (pred_disp)
        gt_dpth = (gt_disp.squeeze().cpu().numpy())


        img_save_pth = os.path.join(args.outdir, ckpt_name) #'figure_paper'#
        if not os.path.isdir(img_save_pth + '/viz'):
            os.makedirs(img_save_pth + '/viz')

        # save for eval
        img_id = inx + 400
        cv2.imwrite('{}/{}_pred.png'.format(img_save_pth, img_id), (pred_dpth * 10000).astype(np.uint16))
        cv2.imwrite('{}/{}_gt.png'.format(img_save_pth, img_id), (gt_dpth * 10000).astype(np.uint16))

        # =========== only need for debug ================
        # err map
        # mask = (gt_dpth > 0)  # .float()
        # err = (np.abs(pred_dpth.clip(0,1.5) - gt_dpth.clip(0, 1.5)) * mask).clip(0, 0.3)
        #
        # cv2.imwrite('{}/viz/{}_err.png'.format(img_save_pth, img_id), err * (255/0.3))

        # pred viz
        # MAX_DISP, MIN_DISP = 1.5, 0
        # # pred_disp = pred_disp.squeeze().detach().cpu().numpy()
        # plt.figure()
        # plt.imshow(pred_disp, vmax=MAX_DISP, vmin=MIN_DISP)  # val2uint8(, MAX_DISP, MIN_DISP)
        # plt.axis('off')
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.savefig('{}/viz/{}_pred_viz.png'.format(img_save_pth, img_id), bbox_inches='tight', pad_inches=0)
        # plt.close()
        #
        # # std viz
        # plt.imshow(std.squeeze().detach().cpu().numpy(), vmax=0.5, vmin=0)  # val2uint8(, MAX_DISP, MIN_DISP)
        # plt.axis('off')
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.savefig('{}/viz/{}_std_viz.png'.format(img_save_pth, img_id,  args.level), bbox_inches='tight', pad_inches=0)
        #
        # for i in range(args.stack_num):
        #     MAX_DISP, MIN_DISP = 1, 0
        #     plt.imshow(focusMap[i].squeeze().detach().cpu().numpy(), vmax=MAX_DISP,
        #                vmin=MIN_DISP, cmap='jet')  # val2uint8(, MAX_DISP, MIN_DISP)
        #     plt.axis('off')
        #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
        #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #     plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        #     plt.margins(0, 0)
        #     plt.savefig('{}/{}_{}_prob_dist.png'.format(img_save_pth, img_id, i), bbox_inches='tight', pad_inches=0)

        # time
        time_list.append('{} {}\n'.format(img_id, ttime))

    print('avgUnc.', std_sum / len(ValImgLoader))
    with open('{}/{}/runtime.txt'.format(args.outdir, ckpt_name), 'w') as f:
        for line in time_list:
            f.write(line)


if __name__ == '__main__':
    main()

