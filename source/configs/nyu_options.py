from configs.base_options import BaseOptions
import argparse

class NYUOptions(BaseOptions):
    def initialize(self):
        parser = BaseOptions.initialize(self)

        # experiment configs
        parser.add_argument('--epochs',      type=int,   default=25)
        parser.add_argument('--max_lr',          type=float, default=1e-4)
        parser.add_argument('--min_lr',          type=float, default=1e-4)
        parser.add_argument('--weight_decay',          type=float, default=5e-2)
        parser.add_argument('--layer_decay',          type=float, default=0.9)
        parser.add_argument('--max_train_dist',          type=float, default=2.0)
        parser.add_argument('--is_blur',type=int,default=1)
        parser.add_argument('--is_kcam',type=int,default=1)
        
        parser.add_argument('--crop_h',  type=int, default=480)
        parser.add_argument('--crop_w',  type=int, default=480)        
        parser.add_argument('--log_dir', type=str, default='./logs')

        # logging options
        parser.add_argument('--val_freq', type=int, default=1)
        parser.add_argument('--save_model', action='store_true')    

        parser.add_argument('--data_path', type=str, default='C:\\Users\\lahir\\data\\kinectmobile\\')
        parser.add_argument('--rgb_dir', nargs="+", default=['refocused_f_50_fdist_2','refocused_f_10_fdist_3'])
        parser.add_argument('--depth_dir', type=str, default='rawDepth')
        parser.add_argument('--resume_from', type=str, default='')
        # parser.add_argument('--resume_from', type=str, default='C:\\Users\\lahir\\Documents\\refocused_f_50_fdist_2.tar')

        parser.add_argument('--eval_trained_rgb_dir', type=str, default='refocused_f_50_fdist_2')
        parser.add_argument('--eval_test_rgb_dir',  nargs="+", default=['refocused_f_75_fdist_2'])
        parser.add_argument('--trained_model', type=str, default="C:\\Users\\lahir\\Documents\\f_50_fdist_2.tar")
        return parser