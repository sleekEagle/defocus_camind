# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/sensors/azure_kinect_recorder.py

import argparse
import open3d as o3d
import os
import glob
import os.path
import numpy as np
from PIL import Image
import simpleaudio 
import time

parser = argparse.ArgumentParser(description='Azure kinect mkv recorder.')
parser.add_argument('--config', type=str, default=r'C:\Users\lahir\code\defocus_camind\source\datacollect\kinect_calib.json',help='input json kinect config')
parser.add_argument('--output', type=str, default='C:\\Users\\lahir\\data\\kinecthands\\',help='output mkv filename')
parser.add_argument('--list',
                    action='store_true',
                    help='list available azure kinect sensors')
parser.add_argument('--device',
                    type=int,
                    default=0,
                    help='input kinect device id')
parser.add_argument('--kinectmode', type=str, nargs='+', help='store kinect depth or color or both',
                    default=['color','depth'])
parser.add_argument('--sensor', type=str, nargs='+', help='store kinect depth or color or both',
                    default=['kinect','mobile'])
parser.add_argument('--n_imgs',
                    type=int,
                    default=2,
                    help='number of images to obtain. -1 for infinity')
args = parser.parse_args()

#create directories if they are not there
if 'kinect' in args.sensor:
    kdir=os.path.join(args.output,'kinect')
    if not os.path.exists(kdir):
        os.makedirs(kdir)
    if 'depth' in args.kinectmode:
        depthdir=os.path.join(kdir,'depth')
        if not os.path.exists(depthdir):
            os.makedirs(depthdir)
    if 'color' in args.kinectmode:
        colordir=os.path.join(kdir,'rgb')
        if not os.path.exists(colordir):
            os.makedirs(colordir)
if 'mobile' in args.sensor:
    adir=os.path.join(args.output,'OpenCamera')
    if not os.path.exists(adir):
        os.makedirs(adir)

#setting up audio clips to play
ROOT_DIR = os.path.dirname(
    os.path.abspath(__name__)
)
audio_path=os.path.join(ROOT_DIR,'audioclips')

camera_obj = simpleaudio.WaveObject.from_wave_file(os.path.join(audio_path,'camera.wav'))
ready_obj = simpleaudio.WaveObject.from_wave_file(os.path.join(audio_path,'ready.wav'))
sad_obj = simpleaudio.WaveObject.from_wave_file(os.path.join(audio_path,'sad.wav'))

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

if 'kinect' in args.sensor:
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
    r = RecorderWithCallback(config, device, filename,True)

#mode : color or depth
def take_kinect_photo(img_n,mode=['color']):    
    
    
    r.run()

    #init reader
    reader = o3d.io.AzureKinectMKVReader()
    reader.open(filename)
    if not reader.is_opened():
        raise RuntimeError("Unable to open file {}".format(filename))
    while not reader.is_eof():
        rgbd = reader.next_frame()
        if args.output is not None and rgbd is not None:
            if 'color' in mode:
                o3d.io.write_image(os.path.join(colordir,str(img_n)+'.jpg'), rgbd.color)
            if 'depth' in mode:
                depthimg=Image.fromarray(np.asarray(rgbd.depth))
                depthimg.save(os.path.join(depthdir,str(img_n)+'.png'))
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
    time.sleep(0.2)
    # out=os.system(adb_path + ' shell ls sdcard/DCIM/OpenCamera/')
    pull_out=os.system(adb_path + ' pull sdcard/DCIM/OpenCamera/ ' + args.output)
    time.sleep(0.2)
    if not pull_out==0:
        return -1
    folder_path =args.output+'OpenCamera'
    file_type = r'\*jpg'
    files = glob.glob(folder_path + file_type)
    max_file = max(files, key=os.path.getctime)
    out=os.rename(max_file,folder_path+'\\'+str(img_n)+'.jpg')
    time.sleep(0.2)
    return out

def files_corrupted():
    play_obj = sad_obj.play()
    play_obj.wait_done()
    print('Directories have different number of files. Please check and correct them.')


current_n=0
while(True):
    #check if all the directories have the same number of files
    n_kinect_rgb,n_kinect_depth,n_android=-1,-1,-1
    try:
        n_kinect_rgb=len(os.listdir(colordir))
    except:
        pass
    try:
        n_kinect_depth=len(os.listdir(depthdir))
    except:
        pass
    try:
        n_android=len(os.listdir(adir))
    except:
        pass

    numlist=[n_kinect_rgb,n_kinect_depth,n_android]
    numlist_selected=[item for item in numlist if item>=0]
    assert len(numlist_selected)>0,'need to use at least one sensor'
    for i,item in enumerate(numlist_selected):
        if i==0:
            first=numlist_selected[0]
            continue
        assert item==first,files_corrupted()
    next_n=first+1

    if 'kinect' in args.sensor:
        take_kinect_photo(next_n,args.kinectmode)
    if 'mobile' in args.sensor:
        take_open_camera_photo(next_n)
    
    play_obj = camera_obj.play()
    play_obj.wait_done()

    current_n+=1
    if current_n==args.n_imgs:
        break
    
    #sleep for a bit. Let the human get ready.
    time.sleep(1)
    # play_obj = ready_obj.play()
    # play_obj.wait_done()
    # time.sleep(1)
    print(str(current_n)+" done.")


    














    


