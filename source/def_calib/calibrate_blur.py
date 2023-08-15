import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

calib_mtx_pth='C:\\Users\\lahir\\data\\calibration\\kinect_calib\\kinect\\k.npy'
dist_mtx_pth='C:\\Users\\lahir\\data\\calibration\\kinect_calib\\kinect\\dist.npy'
photos_dir='C:\\Users\\lahir\\data\\calibration\\kinect_blur\\kinect\\refocused\\'
undist_pth='C:\\Users\\lahir\\data\\calibration\\kinect_blur\\kinect\\refocused_corrected\\'

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


img_path=r'C:\Users\lahir\data\calibration\kinect_blur\kinect\refocused_corrected\0.8.png'
def est_sigmas(img_path):
    gray=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    #detect circle centers
    grid_size=(4,11)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners = cv2.findCirclesGrid(gray, grid_size,None,flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

    if ret:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        img = cv2.drawChessboardCorners(gray.copy(), grid_size, corners2,ret)
        cv2.imshow('I2',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
            selected_sigma=sigma_est[s<1]
            sigma=np.mean(selected_sigma)
            std=np.std(selected_sigma)
            sigma_list.append(sigma)
            sigma_std_list.append(std)
        return sigma_list,sigma_std_list,corners2
    else:
        return -1


#read calibration and distortion matrices
mtx=np.load(calib_mtx_pth)
dist=np.load(dist_mtx_pth)

#read image from the blurred dir
img_pths=os.listdir(photos_dir)
img_pths.sort()

#remove distortions from the images and save them
for pth in img_pths:
    img = cv2.imread(os.path.join(photos_dir,pth))
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(undist_pth,pth[:-4]+'.png'), dst)


img_pths=os.listdir(undist_pth)
img_pths.sort()
sigma_list_list=[]
fdist=[]
center_list=[]
for img in img_pths:
    fd=float(img[:-4])
    fdist.append(fd)
    print('fdist:'+str(fd))
    sigma_list,sigma_std_list,centers=est_sigmas(os.path.join(undist_pth,img))
    sigma_list_list.append(sigma_list)
    center_list.append(centers)

for center in center_list:
    

sig=[]
for item in sigma_list_list:
    sig.append(item[0])


#select the image that is most focused

mean_sigmas=np.array([np.mean(np.array(item)) for item in sigma_list_list])
min_arg=np.argmin(mean_sigmas)






plt.plot(fdist,sig)
plt.show()

kernel_size = 2
kernel = np.ones(kernel_size) / kernel_size
data_convolved = np.convolve(sig, kernel, mode='same')[int(kernel_size/2):]

plt.plot(data_convolved)
plt.show()

