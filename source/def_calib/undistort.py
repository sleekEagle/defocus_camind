import os
import cv2
import numpy as np


calib_mtx_pth='C:\\Users\\lahir\\data\\calibration\\kinect_calib\\kinect\\k.npy'
dist_mtx_pth='C:\\Users\\lahir\\data\\calibration\\kinect_calib\\kinect\\dist.npy'
img_dir='C:\\Users\\lahir\\data\\calibration\\kinect_blur\\kinect\\blur_calib\\f_40\\'

#read calibration and distortion matrices
mtx=np.load(calib_mtx_pth)
dist=np.load(dist_mtx_pth)  

#remove distortions from the images and save them
def undistort(mtx,dist,input_pth,out_pth):
    img_pths=os.listdir(input_pth)
    img_pths=[img for img in img_pths if (img[-3:]=='png' or img[-3:]=='jpg')]

    for pth in img_pths:
        img = cv2.imread(os.path.join(input_pth,pth))
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(out_pth,pth[:-4]+'.png'), dst)

undistort(mtx,dist,'C:\\Users\\lahir\\data\\calibration\\kinect_blur\\kinect2\\kinect\\cameras\\f_50\\refocused\\','C:\\Users\\lahir\\data\\calibration\\kinect_blur\\kinect2\\kinect\\cameras\\f_50\\undist\\')
