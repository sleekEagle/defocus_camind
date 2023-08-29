import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import os
sys.path.append(os.getcwd())
import def_calib.calibrate_cam
import math
from scipy import integrate

grid_size=(4,11)
fdist=2.0
#distance between ajecent circle centers of the calibration pattern
d=14

calib_mtx_pth='C:\\Users\\lahir\\data\\pixelcalib\\telephoto\\k.npy'
dist_mtx_pth='C:\\Users\\lahir\\data\\pixelcalib\\telephoto\\dist.npy'
img_dir='C:\\Users\\lahir\\data\\pixelcalib\\telephoto\\blurcalib\\'


'''
|s2-s1|/s2/(s1-f) * f**2/N/px*out_pix/sensor_pix = k_r * sigma
f**2/N/px*out_pix/sensor_pix/(s1-f) = sigma * s2/|s2-s1|
k/(s1-f) = sigma * s2/|s2-s1|
k_cam = sigma * s2/|s2-s1|

This code estiamtes the k_cam
focal length (fdist or s1 in the equations above) must be constant for all the images taken
'''

#read calibration and distortion matrices
mtx=np.load(calib_mtx_pth)
dist=np.load(dist_mtx_pth)  

blurred_pth=os.path.join(img_dir,'blurred')
focused_pth=os.path.join(img_dir,'focused')

def imshow(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#estimate sigma from an image segment
def est_sigma(img):
    center=(int(img.shape[0]/2),int(img.shape[1]/2))
    explore_dist=int(img.shape[0]/10)
    I_list=[]
    max_list=[]
    for ex in range(-explore_dist,explore_dist):
        data=img[center[0]+ex,:]*-1
        max_list.append(max(data))
        data=(data-min(data))/(max(data)-min(data))
        #where are the falling
        falling=(data<0.95)*1.0
        data_=data*falling
        I=integrate.simpson(data_,np.arange(0,len(data)))
        I_list.append(I)
    #remove outliers
    I_list=np.array(I_list)
    d = np.abs(I_list - np.median(I_list))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    selected_I=I_list[s<1.0]
    I_est=np.mean(selected_I)
    sigma=I_est/np.sqrt(2*math.pi)
    return sigma

def get_centers(gray):
    #detect circle centers
    grid_size=(4,11)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners = cv2.findCirclesGrid(gray, grid_size,None,flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    if ret:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        return corners2
    else:
        return -1
    
#get distances to all coordinated in centers
#assume the points are are asymetric circular patters with 
#d as the longer distance between adjecent circle centers
def get_center_dist(mtx,dist,centers):
    objp=def_calib.calibrate_cam.get_obj_points(d=d)
    ret,rvecs, tvecs = cv2.solvePnP(objp, centers, mtx, dist)
    R,_=cv2.Rodrigues(rvecs)
    objp_t=np.transpose(objp)
    #object points in cam coordinate system
    cam_pos=np.matmul(R,objp_t) + tvecs
    center_dists=np.sqrt(np.sum(np.square(cam_pos),axis=0))
    return center_dists

def get_sigma_list(img):
    centers_=get_centers(img)
    center_dist=get_center_dist(mtx,dist,centers_)
    center_dist=center_dist*1e-3
    # centers_=get_centers(blurred_img)
    x_range=np.max(centers_[:,0,0])-np.min(centers_[:,0,0])
    y_range=np.max(centers_[:,0,1])-np.min(centers_[:,0,1])
    bound=min(x_range,y_range)/max(grid_size)*1.2
    sigma_list=[]
    for k in range(len(centers_)):
        x,y=centers_[k,0,0],centers_[k,0,1]
        seg=img[int(y-bound):int(y+bound),int(x-bound):int(x+bound)]
        #get sigma for this segment
        sigma_=est_sigma(seg)
        sigma_list.append(sigma_)
    sigma_list=np.array(sigma_list)
    return sigma_list,center_dist
    

focused_files=[os.path.join(focused_pth,f) for f in os.listdir(focused_pth) if ((f[-3:]=='png') or (f[-3:]=='jpg'))]
focused_files.sort()
blurred_files=[os.path.join(blurred_pth,f) for f in os.listdir(blurred_pth) if ((f[-3:]=='png') or (f[-3:]=='jpg'))]
blurred_files.sort()
kcam_list=[]
print('number of images = '+str(len(blurred_files)))
for i,file in enumerate(blurred_files):
    # focused_img=cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    blurred_img=cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    focused_img=cv2.imread(focused_files[i],cv2.IMREAD_GRAYSCALE)
    blurred_sigmas,blurred_center_dist=get_sigma_list(blurred_img)
    focused_sigmas,focused_center_dist=get_sigma_list(focused_img)

    blurred_sigmas_=np.sqrt(blurred_sigmas**2 - focused_sigmas**2)

    # print(np.mean(np.array(center_dist)))
    # blurred_kcam=blurred_sigmas*blurred_center_dist/np.abs(blurred_center_dist - fdist)
    # focused_kcam=focused_sigmas*focused_center_dist/np.abs(focused_center_dist - fdist)
    # kcam_=blurred_kcam-focused_kcam*0.59

    kcam_=blurred_sigmas_*blurred_center_dist/np.abs(blurred_center_dist - fdist)
    
    kcam_list=kcam_list+list(kcam_)

print('number of k_s values obtained = '+str(len(kcam_list)))

kcam_list = [item for item in kcam_list if item >0]
#remove outliers from the ks
kcam_list=np.array(kcam_list)
d = np.abs(kcam_list - np.median(kcam_list))
mdev = np.median(d)
s = d/mdev if mdev else np.zeros(len(d))
selected_kcams=kcam_list[s<0.5]
print('number of k_s values retained after outlier removal = '+str(len(selected_kcams)))
kcam_est=np.mean(selected_kcams)
print('k_cam_est = '+str(kcam_est))


# f = np.array([10,20,25,30,40,50])
# ks_est=[2.4,3.3,4.3,5.4,8.8,13.0]
# x=((f*1e-3)**2)/(2-f*1e-3)

# plt.plot(x,ks_est,'b-')
# plt.show()


def gaussian(x,sigma):
    g=np.exp(-1*(x)**2/(2*sigma**2))
    return g

def fit_sigma(data,guess=12):
    #filter the signal
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved = np.convolve(data, kernel, mode='same')[int(kernel_size/2):]
    #get derivative
    diff=np.abs(np.diff(data_convolved))
    thr=np.min(np.argwhere(diff>0.003))
    clean_data=data_convolved[thr:]
    initial_guess=[guess]
    x=np.linspace(0,len(clean_data),len(clean_data))
    popt, pcov = curve_fit(gaussian, x, clean_data, p0=initial_guess)
    sigma_est=popt[0]
    return sigma_est



def est_sigmas(img_path):
    gray=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    #detect circle centers
    grid_size=(4,11)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners = cv2.findCirclesGrid(gray, grid_size,None,flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

    if ret:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        img = cv2.drawChessboardCorners(gray.copy(), grid_size, corners2,ret)
        # cv2.imshow('I2',seg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        x_range=np.max(corners2[:,0,0])-np.min(corners2[:,0,0])
        y_range=np.max(corners2[:,0,1])-np.min(corners2[:,0,1])
        bound=min(x_range,y_range)/max(grid_size)*1.2

        sigma_list=[]
        sigma_std_list=[]
        for i in range(len(corners2)):
            x,y=corners2[i,0,0],corners2[i,0,1]
            seg=gray[int(y-bound):int(y+bound),int(x-bound):int(x+bound)]

           
            detected_circles = cv2.HoughCircles(seg, 
                            cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                        param2 = 20, minRadius = 1, maxRadius = 40)
            detected_circles = np.uint16(np.around(detected_circles))

            pt=detected_circles[0,0, :]
            a, b, r = pt[0], pt[1], pt[2]
            sample_positions_x=np.arange(int(a-r/2),int(a+r/2),1)
            sample_positions_y=np.arange(int(b-r/2),int(b+r/2),1)

            sigma_est=[]
            #slice parallel to the x axis
            for x_pos in sample_positions_x:
                x_vals=seg[x_pos,:]
                x_vals=x_vals*-1
                scaled=(x_vals-min(x_vals))/(max(x_vals)-min(x_vals))
                data1=scaled[int(len(scaled)/2):]
                sigma_est1=fit_sigma(data1,guess=10)
                sigma_est.append(sigma_est1)
                data2=np.flip(scaled[:int(len(scaled)/2)])
                sigma_est2=fit_sigma(data2,guess=10)
                sigma_est.append(sigma_est2)
            #slice parallel to the y axis
            for y_pos in sample_positions_y:
                y_vals=seg[y_pos,:]
                y_vals=y_vals*-1
                scaled=(y_vals-min(y_vals))/(max(y_vals)-min(y_vals))
                data1=scaled[int(len(scaled)/2):]
                sigma_est1=fit_sigma(data1,guess=10)
                sigma_est.append(sigma_est1)
                data2=np.flip(scaled[:int(len(scaled)/2)])
                sigma_est2=fit_sigma(data2,guess=10)
                sigma_est.append(sigma_est2)
            #reject outliers
            #from https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
            sigma_est=np.array(sigma_est)
            d = np.abs(sigma_est - np.median(sigma_est))
            mdev = np.median(d)
            s = d/mdev if mdev else np.zeros(len(d))
            selected_sigma=sigma_est[s<0.5]
            sigma=np.mean(selected_sigma)
            std=np.std(selected_sigma)
            sigma_list.append(sigma)
            sigma_std_list.append(std)
        return sigma_list,sigma_std_list,corners2
    else:
        return -1
    

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

#get sigmans and the center coordinates from all the images in a directory
def get_sigmas(img_pth):
    img_pths=os.listdir(img_pth)
    img_pths.sort()
    sigma_list_list=[]
    center_list=[]
    for img in img_pths:
        sigma_list,sigma_std_list,centers=est_sigmas(os.path.join(img_pth,img))
        sigma_list_list.append(sigma_list)
        center_list.append(centers)
    sigma_list_list=np.array(sigma_list_list)
    center_list=np.array(center_list)
    return sigma_list_list,center_list

#get distances to all coordinated in centers
#assume the points are are asymetric circular patters with 
#d as the longer distance between adjecent circle centers
def get_center_dist(mtx,dist,centers):
    objp=def_calib.calibrate_cam.get_obj_points(d=151)
    ret,rvecs, tvecs = cv2.solvePnP(objp, centers, mtx, dist)
    R,_=cv2.Rodrigues(rvecs)
    objp_t=np.transpose(objp)
    #object points in cam coordinate system
    cam_pos=np.matmul(R,objp_t) + tvecs
    center_dists=np.sqrt(np.sum(np.square(cam_pos),axis=0))
    return center_dists
