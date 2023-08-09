import open3d as o3d
import scipy.io
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt


#***********read camera parameters****************#
#pose of camera2 (mobile) wrt camera1 (kinect)
cam2pose_pt=r'C:\Users\lahir\data\calib_images\cam2pose.txt'
cam2pose = open(cam2pose_pt)
cam2pose = np.loadtxt(cam2pose, delimiter=",")
cam2pose=o3d.core.Tensor(cam2pose)

# cam2pose[0:3,-1]=cam2pose[0:3,-1]

#camera1 intrincics
cam1K_pth=r'C:\Users\lahir\data\calib_images\K1.txt'
cam1K=open(cam1K_pth)
cam1intr = np.loadtxt(cam1K, delimiter=",")
cam1intr=o3d.core.Tensor(cam1intr)

#camera1 intrincics
cam2K_pth=r'C:\Users\lahir\data\calib_images\K2.txt'
cam2K=open(cam2K_pth)
cam2intr = np.loadtxt(cam2K, delimiter=",")
cam2intr=o3d.core.Tensor(cam2intr)


depth = o3d.t.io.read_image(r'C:\Users\lahir\data\kinectmobile\kinect\depth\1.png')
ptc = o3d.t.geometry.PointCloud.create_from_depth_image(depth,
                                                            cam1intr,
                                                            depth_scale=1.0,
                                                            depth_max=10.0)




T = np.eye(4)
pcd_t = ptc.clone().transform(T)


pose = np.array([[0.997558 ,0.00550184 ,0.0696318 ,-56.7901], [-0.00676462 ,0.999817 ,0.0179123 ,62.6944],
                                 [-0.0695205 ,-0.0183396 ,0.997412 ,24.2925],[0,0,0,1]])
pcd_t = ptc.clone().transform(pose)



intrinsic = o3d.core.Tensor([[922.0, 0, 959.19], [0, 922.3, 551.44],
                                 [0, 0, 1]])

intrinsic = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6],
                                 [0, 0, 1]])

depth_reproj = pcd_t.project_to_depth_image(1920,
                                              1080,
                                              intrinsic,
                                              depth_scale=5000.0,
                                              depth_max=10.0)








plt.imshow(depth_reproj)
plt.show()




o3d.visualization.draw([ptc_trans])

depth_reproj = ptc.project_to_depth_image(width=1920,heigth=1080,
                                              depth_scale=5000.0,
                                              depth_max=10.0)


import matplotlib.pyplot as plt
plt.imshow(depth_reproj)
plt.show()



ptc_trans=ptc.transform(cam2pose)





fig, axs = plt.subplots(1, 2)
axs[0].imshow(np.asarray(depth.to_legacy()))
axs[1].imshow(np.asarray(depth_reproj.to_legacy()))
plt.show()










#transform coordinate system to camera 2 from camera 1 (from kinect to mobile phone)
ptc_trans=ptc.transform(cam2pose)
ptc_trans.project_to_depth_image





#project back to the second camera
ptc_np=np.asarray(ptc_trans.points)
depth=np.linalg.norm(ptc_np,axis=1)
imgpts=np.matmul(cam2intr,np.transpose(ptc_np))
imgpts=imgpts/imgpts[-1:,:]
imgpts_round=np.round(imgpts)
img=np.ones((1080,1920))*-1
for i in range(imgpts_round.shape[1]):
    item=imgpts_round[:,i]
    xcoord,ycoord=round(item[0]),round(item[1])
    if xcoord>0 and xcoord<1920 and ycoord>0 and ycoord<1080:
        # print('xcord:'+str(xcoord))
        # print('ycord:'+str(ycoord))
        #we need to project only the closest item in the same line of sight
        current_d=img[ycoord,xcoord]
        new_d=depth[i]
        if current_d==-1 and new_d>0:
            img[ycoord,xcoord]=new_d
        elif new_d < current_d and new_d>0:
            img[ycoord,xcoord]=new_d
img[img==-1]=0
img=img.astype(np.uint16)

#fill in the missing values
mask=img==0
mask=mask.astype(np.uint8)
dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
cv2.imwrite(r'C:\Users\lahir\data\kinectmobile\OpenCamera\processed\1_filled.png',dst)


import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()

o3d.visualization.draw_geometries([ptc],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
















