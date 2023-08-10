path='C:\Users\lahir\data\kinectmobile';
kinectdepth_pth=strcat(path,'\kinect\depth\');
kinectrgb_pth=strcat(path,'\kinect\rgb\');
mobile_rgb=strcat(path,'\OpenCamera\');
mobile_processed_pth=strcat(path,'\mobileprocessed');
mobile_rgb_imgs=dir(strcat(mobile_rgb,'*.jpg'));


%loasd the calibration data 
load('C:\Users\lahir\data\calib_images\calib.mat')

%%
for k=1:size(mobile_rgb_imgs,1)
    s=split(mobile_rgb_imgs(k).name,'.');
    n=s{1};
    mobile_rgb_file=fullfile(mobile_rgb_imgs(k).folder,mobile_rgb_imgs(k).name);
    kinectname=strcat(n,'.png');
    kinect_depth_file=fullfile(kinectdepth_pth,kinectname);

    %undistort images
    depthI=imread(kinect_depth_file);
    %undistDepth=undistortImage(depthI,stereoParams.CameraParameters1);
    mobileI=imread(mobile_rgb_file);
    %undistMobile=undistortImage(mobileI,stereoParams.CameraParameters2);
    
    img_tr=transform_depth_img(stereoParams,depthI,mobileI);
    img_tr=uint16(img_tr);
    out_file=fullfile(mobile_processed_pth,kinectname);
    imwrite(img_tr,out_file,BitDepth=16);
end



