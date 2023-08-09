load('C:\Users\lahir\data\calib_images\calib.mat')
kinect_depth='C:\Users\lahir\data\kinectmobile\kinect\depth\1.png';
mobile_rgb='C:\Users\lahir\data\kinectmobile\OpenCamera\1.jpg';
img_tr=transform_depth_img(stereoParams,ckinect_depth,mobile_rgb);