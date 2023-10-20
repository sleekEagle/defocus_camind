function [] = blur_images(f,focus,dsigma,saveDepth)

%{
f = focal length in mm
focus = focus distance in m
dsigma = sigma used to mean filter the depth image before refocuseing 
        the RGB images
    sigma is used in imgaussfilt(img_,dsigma)
    this is done if only dsigma>0
%}

disp(['focal length = ',num2str(f),' mm'])
disp(['focus distance = ',num2str(focus),' m'])
out_dir=strcat('D:\data\nyu_depth_v2\ours\','f_',num2str(f),'_fdist_',num2str(focus),'_dsigma_',num2str(dsigma),'\');
depth_out_dir=strcat('D:\data\nyu_depth_v2\ours\','f_',num2str(f),'_fdist_',num2str(focus),'_depth\');
create_dir(out_dir);
if saveDepth
    create_dir(depth_out_dir);
end
f=f*1e-3;
px=36*1e-6;
N=1.0;  
%focus=2.0;
depth_step=0.005;

%original focal legth in pixels (of the camera images were taken with)
f_pix_original=906.84;
%focal length in pixels of the given camera
f_pix=f/px;
%the scale factor; by what factor should we scale the original image 
%that agrees with the angle of view of the given camera
scale=f_pix/f_pix_original;
disp(['scale = ',num2str(scale)]);
%original image sizes 
img_w=480;
img_h=640;

rgb_dir='D:\data\nyu_depth_v2\rgb_f_0_fdist_0\';
depth_dir='D:\data\nyu_depth_v2\filledDepth\';
raw_depth_dir='D:\data\nyu_depth_v2\rawDepth\';

rgb_files=dir(rgb_dir);
depth_files=dir(depth_dir);

for i=3:(length(rgb_files))
    disp(rgb_files(i).name)
    %read image
    rgb_path=strcat(rgb_dir,rgb_files(i).name);
    depth_path=strcat(depth_dir,depth_files(i).name);
    raw_depth_path=strcat(raw_depth_dir,depth_files(i).name);

    out_path=strcat(out_dir,depth_files(i).name);
    depth_out_path=strcat(depth_out_dir,depth_files(i).name);
    
    rgb=double(imread(rgb_path));
    depth=imread(depth_path);
    raw_depth=imread(raw_depth_path);

    if dsigma > 0
        depth=imgaussfilt(depth,dsigma);
        %imshow(depth,[0,3650])
    end
    
    %convert depth into meters
    depth=double(double(depth)/1000.0);

    %resize depth and RGB images
    raw_depth=imresize(raw_depth,scale);
    rgb=imresize(rgb,scale);
    depth=imresize(depth,scale);

    d_values=(min(depth(depth>0))-depth_step):depth_step:(max(depth(depth>0))+depth_step);
    refocused = zeros(size(rgb));

    for k=1:length(d_values)
        if k==length(d_values)
            d=(d_values(k)+depth_step*0.5);
        else
            d=0.5*(d_values(k)+d_values(k+1));
        end
     
        sigma=abs(d-focus).*(1./d) / (focus-f) * f^2/N *0.3 /px;
       
        z_=zeros(size(depth));
        if k==length(d_values)
            z_((depth>=d_values(k)))=1;
        else
            z_((depth<(d_values(k+1))) & (depth>=(d_values(k))))=1;
        end
        %dialate z_ hopefully to cover zinvalid depth areas
        %z_=imdilate(z_,se);
        z_=repmat(z_,1,1,3);
        img_=rgb.*z_;
        %imshow(img_)
        refocused_=imgaussfilt(img_,sigma);
        %refocused_=z_*sigma;
        %disp(z_(z_>0))
        %refocused_=imdilate(refocused_,se);
        refocused=refocused+refocused_;
        %imshow(uint8(refocused))
    end
    imwrite(uint8(refocused),out_path);
    if saveDepth
        %convert depth back to mm
        imwrite(uint16(raw_depth),depth_out_path,'BitDepth',16);
    end
end
end
 

function [] = create_dir(dir_path)
    if(exist(dir_path)~=7)
        mkdir(dir_path)
    else
        display([dir_path  'already exists'])
    end
end



