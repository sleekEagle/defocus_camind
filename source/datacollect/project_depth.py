import open3d as o3d
import scipy.io
import numpy as np
import cv2

#***********read camera parameters****************#
#pose of camera2 (mobile) wrt camera1 (kinect)
cam2pose_pt=r'C:\Users\lahir\data\calib_images\cam2pose.txt'
cam2pose = open(cam2pose_pt)
cam2pose = np.loadtxt(cam2pose, delimiter=",")
cam2pose[0:3,-1]=cam2pose[0:3,-1]

#camera1 intrincics
cam1K_pth=r'C:\Users\lahir\data\calib_images\K1.txt'
cam1K=open(cam1K_pth)
cam1intr = np.loadtxt(cam1K, delimiter=",")

#camera1 intrincics
cam2K_pth=r'C:\Users\lahir\data\calib_images\K2.txt'
cam2K=open(cam2K_pth)
cam2intr = np.loadtxt(cam2K, delimiter=",")


depth_raw = o3d.io.read_image(r'C:\Users\lahir\data\kinectmobile\kinect\depth\1.png')
intr=o3d.camera.PinholeCameraIntrinsic()
intr.set_intrinsics(width=1920,height=1080,fx=cam1intr[0,0],fy=cam1intr[1,1],cx=cam1intr[0,-1],cy=cam1intr[1,-1] )
ptc=o3d.geometry.PointCloud.create_from_depth_image(depth_raw,intrinsic=intr,
                                                 depth_scale=1.0, depth_trunc=3000.0, stride=1)

#transform coordinate system to camera 2 from camera 1 (from kinect to mobile phone)
ptc_trans=ptc.transform(cam2pose)

#project back to the second camera
ptc_np=np.asarray(ptc_trans.points)
depth=np.linalg.norm(ptc_np,axis=1)
imgpts=np.matmul(cam2intr,np.transpose(ptc_np))
imgpts=imgpts/imgpts[-1:,:]
imgpts_round=np.round(imgpts)
img=np.zeros((1080,1920))
for i in range(imgpts_round.shape[1]):
    item=imgpts_round[:,i]
    xcoord,ycoord=round(item[0]),round(item[1])
    if xcoord>0 and xcoord<1920 and ycoord>0 and ycoord<1080:
        # print('xcord:'+str(xcoord))
        # print('ycord:'+str(ycoord))
        img[ycoord,xcoord]=depth[i]

img=img.astype(np.uint16)

#fill in the missing values
mask=img==0
mask=mask.astype(np.uint8)
dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
cv2.imwrite(r'C:\Users\lahir\data\kinectmobile\OpenCamera\processed\1_filled.png',img)


import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()

o3d.visualization.draw_geometries([ptc_trans],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])









