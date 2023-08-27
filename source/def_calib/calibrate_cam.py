import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

photos_dir='C:\\Users\\lahir\\data\\pixelcalib\\calib\\'
out_path='C:\\Users\\lahir\\data\\pixelcalib\\'

def get_reprojection_errors(objpoints,imgpoints,rvecs,tvecs,mtx,dist):
    errors=[]
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        errors.append(error)
    errors=np.array(errors)
    print( "mean error: {}".format(np.mean(errors)))
    return errors

'''
calibrate with an asymetric circular grid.
d : distance between two adjecent circles (the longer distance) in mm
'''
def get_obj_points(d=42.5,grid_size=(4,11)):
    objp = np.zeros((grid_size[0]*grid_size[1], 3), np.float32)
    #distance between two circle centers in mm 
    for i in range(grid_size[1]):
        for j in range(grid_size[0]):
            indx=grid_size[0]*i+j
            x_value=i*d/2
            y_value=j*d + d/2*(i%2)
            objp[indx,:]=[x_value,y_value,0]
    return objp

def calibrate(photos_dir,d=42.5,grid_size=(4,11)):
    #calibrating with the asymetric circular grid 
    objp=get_obj_points(d,grid_size)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    found=0

    img_pths=os.listdir(photos_dir)
    for i,item in enumerate(img_pths):
        print(str(i)+' done')
        gray=cv2.imread(os.path.join(photos_dir,item),cv2.IMREAD_GRAYSCALE)
        #detect circle centers
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ret, corners = cv2.findCirclesGrid(gray, grid_size,None,flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            im_with_keypoints = cv2.drawChessboardCorners(gray, (4,11), corners2, ret)
            found += 1
            cv2.imshow("img", im_with_keypoints) # display
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    print('num good images:'+str(found))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs,objpoints,imgpoints,gray.shape[::-1]

#calibrate the first round
if __name__ == "__main__":
    ret, mtx, dist, rvecs,tvecs,objpoints,imgpoints,imshape=calibrate(photos_dir,d=29)
    errors=get_reprojection_errors(objpoints,imgpoints,rvecs,tvecs,mtx,dist)

    thr_error=0.2
    valid_args=np.argwhere(errors<thr_error)
    #recalibrate with the selected images
    objpoints_selected=[pt for i,pt in enumerate(objpoints) if i in valid_args]
    imgpts_selected=[pt for i,pt in enumerate(imgpoints) if i in valid_args]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_selected, imgpts_selected,imshape, None, None)
    errors=get_reprojection_errors(objpoints_selected,imgpts_selected,rvecs,tvecs,mtx,dist)

    #save the calibration matrices
    np.save(os.path.join(out_path,'k.npy'),mtx)
    np.save(os.path.join(out_path,'dist.npy'),dist)


















