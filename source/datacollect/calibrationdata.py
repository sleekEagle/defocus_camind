# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/sensors/azure_kinect_recorder.py

import argparse
import datetime
import open3d as o3d
import os
import glob
import os.path
import time

parser = argparse.ArgumentParser(description='Azure kinect mkv recorder.')
parser.add_argument('--config', type=str, default=r'C:\Users\lahir\code\defocus_camind\source\datacollect\kinect_calib.json',help='input json kinect config')
parser.add_argument('--output', type=str, default='C:\\Users\\lahir\\data\\calib_images\\',help='output mkv filename')
parser.add_argument('--list',
                    action='store_true',
                    help='list available azure kinect sensors')
parser.add_argument('--device',
                    type=int,
                    default=0,
                    help='input kinect device id')
parser.add_argument('-a',
                    '--align_depth_to_color',
                    action='store_true',
                    help='enable align depth image to color')
args = parser.parse_args()

class RecorderWithCallback:

    def __init__(self, config, device, filename, align_depth_to_color):
        # Global flags
        self.flag_exit = False
        self.flag_record = False
        self.filename = filename

        self.align_depth_to_color = align_depth_to_color
        self.recorder = o3d.io.AzureKinectRecorder(config, device)
        if not self.recorder.init_sensor():
            raise RuntimeError('Failed to connect to sensor')
        

    def run(self):
        print("Recorder initialized.")
        nframes=0
        while nframes<1:
            if not self.recorder.is_record_created():
                if self.recorder.open_record(self.filename):
                    self.flag_record = True
            rgbd = self.recorder.record_frame(self.flag_record,
                                              self.align_depth_to_color)
            if rgbd is None:
                continue
            nframes+=1

        self.recorder.close_record()


def take_kinect_photo(img_n):    
    if args.list:
        o3d.io.AzureKinectSensor.list_devices()
        exit()

    if args.config is not None:
        config = o3d.io.read_azure_kinect_sensor_config(args.config)
    else:
        config = o3d.io.AzureKinectSensorConfig()

    filename=os.path.join(args.output,'recording.mkv')
    print('Prepare writing to {}'.format(filename))

    device = args.device
    if device < 0 or device > 255:
        print('Unsupported device id, fall back to 0')
        device = 0

    r = RecorderWithCallback(config, device, filename,
                             args.align_depth_to_color)
    r.run()

    #init reader
    reader = o3d.io.AzureKinectMKVReader()
    reader.open(filename)
    if not reader.is_opened():
        raise RuntimeError("Unable to open file {}".format(filename))
    while not reader.is_eof():
        rgbd = reader.next_frame()
        if args.output is not None and rgbd is not None:
            o3d.io.write_image(os.path.join(args.output,"kinect",str(img_n)+'.jpg'), rgbd.color)
            break
    reader.close()


######################################################################
'''
take android photo
1. open the open camera app
2. set focal distance to 2m
3. zoom to 1x
tap x=544 y=2160
'''
########################################################################
def take_open_camera_photo(img_n):
    adb_path=r'C:\Users\lahir\adb\platform-tools_r33.0.3-windows\platform-tools\adb.exe'
    #remove all the images in the open camera directory
    out=os.system(adb_path+' shell rm sdcard/DCIM/OpenCamera/.*') 
    out=os.system(adb_path+' shell rm sdcard/DCIM/OpenCamera/*')
    #take a photo
    os.system(adb_path + ' shell input tap 2206 515')
    out=os.system(adb_path + ' shell ls sdcard/DCIM/OpenCamera/')
    pull_out=os.system(adb_path + ' pull sdcard/DCIM/OpenCamera/ ' + args.output)
    if not pull_out==0:
        return -1
    folder_path =args.output+'OpenCamera'
    file_type = r'\*jpg'
    files = glob.glob(folder_path + file_type)
    max_file = max(files, key=os.path.getctime)
    out=os.rename(max_file,folder_path+'\\'+str(img_n)+'.jpg')
    return out


n_kinect=len(os.listdir(os.path.join(args.output,'kinect')))
n_android=len(os.listdir(os.path.join(args.output,'OpenCamera')))
assert n_kinect==n_android , "image directories are corrupted. Retake images"
next_n=n_android+1

take_kinect_photo(next_n)
take_open_camera_photo(next_n)







    


