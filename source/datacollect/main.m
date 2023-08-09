load('C:\Users\lahir\data\calib_images\calib.mat')
kinect_depth='C:\Users\lahir\data\kinectmobile\kinect\depth\3.png';
mobile_rgb='C:\Users\lahir\data\kinectmobile\OpenCamera\3.jpg';
img_tr=transform_depth_img(stereoParams,kinect_depth,mobile_rgb);
img_tr=uint16(img_tr);
imwrite(img_tr,'C:\Users\lahir\data\kinectmobile\OpenCamera\processed\3.png',BitDepth=16)