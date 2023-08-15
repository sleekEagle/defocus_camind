import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import def_calib.calibrate_cam
from sklearn.linear_model import LinearRegression,RANSACRegressor

calib_mtx_pth='C:\\Users\\lahir\\data\\calibration\\kinect_calib\\kinect\\k.npy'
dist_mtx_pth='C:\\Users\\lahir\\data\\calibration\\kinect_calib\\kinect\\dist.npy'
img_dir='C:\\Users\\lahir\\data\\calibration\\kinect_blur\\kinect\\blur_calib\\f_40\\'

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

def get_centers(img_path):
    gray=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    #detect circle centers
    grid_size=(4,11)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners = cv2.findCirclesGrid(gray, grid_size,None,flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    if ret:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        return corners2
    else:
        return -1

def est_sigmas(img_path):
    gray=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    #detect circle centers
    grid_size=(4,11)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners = cv2.findCirclesGrid(gray, grid_size,None,flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

    if ret:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # img = cv2.drawChessboardCorners(gray.copy(), grid_size, corners2,ret)
        # cv2.imshow('I2',img)
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
            selected_sigma=sigma_est[s<1]
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
    img_pths=[img for img in img_pths if img[-3:]=='jpg']

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


#read calibration and distortion matrices
mtx=np.load(calib_mtx_pth)
dist=np.load(dist_mtx_pth)  

blurred_pth=os.path.join(img_dir,'fdist2')
focused_pth=os.path.join(img_dir,'focused')
undist_pth=os.path.join(img_dir,'undistorted')
undist_blurred_pth=os.path.join(undist_pth,'fdist2')
undist_focused_pth=os.path.join(undist_pth,'focused')

# undistort(mtx,dist,blurred_pth,undist_blurred_pth)
# undistort(mtx,dist,focused_pth,undist_focused_pth)

sigma_list_list,_=get_sigmas(undist_blurred_pth)
#get center coordinates of images
foc_pth=os.listdir(undist_focused_pth)
foc_pth.sort()
center_list=[]
for p in foc_pth:
    c=get_centers(os.path.join(undist_focused_pth,p))
    center_list.append(c)

#get the distances to the circle centers of all images and all circles
center_dists=[]
for i in center_list:
    c=get_center_dist(mtx,dist,center_list[i])
    center_dists.append(c)

'''
k * (s1-f) = |s1-s2|/(s2*sigma)
'''
#estimate k*(s1-f)
#iterate through images
ks_list=[]
for i in range(len(center_dists)):
    s2=center_dists[i]/1000.0
    s1=2.0
    sigma=sigma_list_list[i]
    ks=np.abs(s1-s2)/(s2*sigma)
    ks_list.append(ks)

ks_list=np.array(ks_list)
#remove outliers
d = np.abs(ks_list - np.median(ks_list))
mdev = np.median(d)
s = d/mdev if mdev else np.zeros(len(d))
selected_ks=ks_list[s<1]
ks_est=np.mean(selected_ks)
print("ks_est:"+str(ks_est))

'''
f 40 ks_est = 0.09291
f 20 ks_est = 0.17961
'''

0.092914/0.179615

((2-40e-3)/(40**2))/((2-20e-3)/(20**2))




















# #select the image that is most focused
# mean_sigmas=np.array([np.mean(np.array(item)) for item in sigma_list_list])
# min_arg=np.argmin(mean_sigmas)
# focused_img=img_pths[min_arg]
# focused_centers=center_list[min_arg]




# '''
# for infinity-focused image
# 1/s2 = k*sigma
# '''
# # estimate k from inf focused image
# infarg=np.argmax(fdist_list)
# inf_sigmas=sigma_list_list[infarg]
# k_est=(center_dists/1000.0*inf_sigmas)
# #remove outliers
# d = np.abs(k_est - np.median(k_est))
# mdev = np.median(d)
# s = d/mdev if mdev else np.zeros(len(d))
# selected_k=k_est[s<1]
# k_est=np.mean(selected_k)



# img_pths=os.listdir(inf_pth)
# sig_list=[]
# f_list=[]
# lambdas=[]
# for img in img_pths:
#     sigma_list,sigma_std_list,corners2=est_sigmas(os.path.join(inf_pth,img))
#     sigma_list=np.array(sigma_list)
#     sig_list.append(sigma_list)
#     s1=2.0
#     f_list.append(int(img.split('_')[1]))
#     lambda_=np.mean(np.abs(s1-center_dists)/center_dists/sigma_list)
#     lambdas.append(lambda_)

# ((2-25e-3)/25**2)/((2-30e-3)/30**2)
# lambdas[0]/lambdas[1]

# vals=[s[0] for s in sig_list]
# plt.plot(f_list,vals,'bo')
# plt.show()
# sig_list=np.array(sig_list)
# sig_means=[np.mean(item) for item in sig_list]
# plt.plot(f_list,sig_means,'bo')
# plt.show()

# sig_means[0]/sig_means[1]



# #estimate k from a given inf image
# inf_img_path=r'C:\Users\lahir\data\calibration\kinect_blur\kinect\refocused_corrected\f_dist_2\f_.png'

# def est_k(inf_img_path):
#     inf_sigmas,sigma_std_list,centers=est_sigmas(inf_img_path)
#     inf_sigmas=np.array(inf_sigmas)
#     k_est=(center_dists/1000.0*inf_sigmas)
#     #remove outliers
#     d = np.abs(k_est - np.median(k_est))
#     mdev = np.median(d)
#     s = d/mdev if mdev else np.zeros(len(d))
#     selected_k=k_est[s<1]
#     k_est=np.mean(selected_k)
#     return k_est

# est_k('C:\\Users\\lahir\\data\\calibration\\kinect_blur\\kinect\\refocused\\others\\corrected\\f_20_inf.png')



# #estimate focal length f from the non-infinity focused images
# '''
# f = s1 - |s1-s2|*k/(*sigma*s2)
# '''
# f_est_list=[]
# for i in range(fdist_list.shape[0]):
#     if i==infarg or i<8:
#         continue
#     s1=fdist_list[i]
#     for j in range(focused_centers.shape[0]):
#         s2=center_dists[j]/1000.0
#         sig_k=sigma_list_list[i][j]
#         sig_inf=sigma_list_list[infarg][j]
#         est=s1-np.abs(s1-s2)/s2 * sig_inf/sig_k
#         print(est)
        
    
# fdists=fdist_list[fdist_list<fdist_list[infarg]]






# s=[item [0] for item in sigma_list_list]
# plt.plot(fdist_list[:-1],s[:-1])
# plt.show()














# s2=np.mean(center_dists)/1000.0
# fdist_list=np.array(fdist_list)
# sig=[np.mean(np.array(item)) for item in sigma_list_list]
# y=np.abs(fdist_list-s2)/s2/(fdist_list-40e-3) 
# plt.plot(fdist_list,sig,'bo')
# plt.show()

# plt.plot(fdist_list,sig,'bo')
# plt.plot(fdist_list,y,'bo')
# plt.show()

# plt.plot(sig[6:],y[6:],'bo')
# plt.show()

# y[6:]/sig[6:]
    





# #perform linear regression to find focal length and k
# #indendent variables
# '''
# equation: |s1-s2|/(s2*sigma) = k * s1  - k*f
#               y              =   m*x   +  c

# estimate m and c
# '''
# b_list,sig_list,s1_list=[],[],[]
# for i in range(len(img_pths)):
#     if i<8:
#         continue
#     s1=fdist_list[i]
#     for j in range(focused_centers.shape[0]):
#         # if j>0: break
#         #get s2 in meters
#         s2=center_dists[j]/1000.0
#         sig=sigma_list_list[i][j]
#         b_list.append(abs(s1-s2)/(s2*sig))
#         sig_list.append(sig)
#         s1_list.append(s1)


# k_est,f_est=0.001,40e-3

# def blur_curve_f(X,f):
#     val=k_est*(X-f)
#     return val
# def blur_curve_k(X,k):
#     val=k*(X-f_est)
#     return val

# for i in range(10):
#     popt, pcov=curve_fit(blur_curve_k,np.array(s1_list),np.array(b_list),p0=[k_est])
#     k_est=popt[0]
#     popt, pcov=curve_fit(blur_curve_f,np.array(s1_list),np.array(b_list),p0=[f_est])
#     f_est=popt[0]
#     print(k_est,f_est)









# global f_est,k_est
# def blur_curve_f(X,f):
#     b,s1=X[:,0],X[:,1]
#     val=b/(s1-f)*k_est
#     return val

# def blur_curve_k(X,k):
#     b,s1=X[:,0],X[:,1]
#     val=b/(s1-f_est)*k
#     return val

# b_ar=np.expand_dims(np.array(b_list),axis=1)
# s1_ar=np.expand_dims(np.array(s1_list),axis=1)
# X=np.concatenate((b_ar,s1_ar),axis=1)

# f_est=40e-3
# k_est=46
# for i in range(100):
#     popt, pcov=curve_fit(blur_curve_k,X,np.array(sig_list),p0=[k_est])
#     k_est=popt[0]
#     popt, pcov=curve_fit(blur_curve_f,X,np.array(sig_list),p0=[f_est])
#     f_est=popt[0]
#     print((k_est,f_est))





# plt.plot(x_list,y_list,'bo')
# plt.show()

# y=np.array(y_list)
# x=np.expand_dims(np.array(x_list),axis=1)
# x=np.array(x_list)

# plt.plot(x,y,'bo')
# plt.show()

# reg = LinearRegression().fit(x, y)
# r_sq = reg.score(x, y)
# print(f"coefficient of determination: {r_sq}")
# m=reg.coef_
# c=reg.intercept_




# slope, intercept = np.polyfit(x, y, 1)
# abline_values = [slope * i + intercept for i in x]

# plt.plot(x, y, '--')
# plt.plot(x, abline_values, 'b')
# plt.show()

# m=reg.coef_
# c=reg.intercept_


# model.get_params()
# import math
# math.degrees(np.arctan(0.05))






# plt.plot(fdist,sig)
# plt.show()

# kernel_size = 2
# kernel = np.ones(kernel_size) / kernel_size
# data_convolved = np.convolve(sig, kernel, mode='same')[int(kernel_size/2):]

# plt.plot(data_convolved)
# plt.show()


#  x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
# y = np.array([5, 20, 14, 32, 22, 38])




# #testing
# # seg_path='C:\\Users\\lahir\\data\\calibration\\kinect_blur\\kinect\\seg\\'
# # img_pths=os.listdir(seg_path)
# # img_pths.sort()

# # for i,p in enumerate(img_pths):
# #     if i<5:
# #         continue
# #     img = cv2.imread(os.path.join(seg_path,p))
# #     h,w,_=img.shape
# #     line=img[int(h/2),:]
# #     plt.plot(line)
# #     if i==8:
# #         break

# # plt.show()

