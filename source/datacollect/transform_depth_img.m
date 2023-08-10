function final = transform_depth_img(stereoParams,depthImage,mobileImg)
    %loading calibration data
    
    cam1intr=stereoParams.CameraParameters1.K;
    focalLength    = [cam1intr(1,1) cam1intr(1,1)]; 
    principalPoint = [cam1intr(1,3) cam1intr(2,3)];
    imsize=stereoParams.CameraParameters1.ImageSize;
    imageSize      = [imsize(1) imsize(2)];
    intrinsics = cameraIntrinsics(focalLength,principalPoint,imageSize);
    depthScaleFactor = 1;
    maxCameraDepth   = 10;
    ptc=pcfromdepth(depthImage,depthScaleFactor,intrinsics);
    
    %transform into camera2 coord system
    tform = rigidtform3d(stereoParams.PoseCamera2.A);
    ptc_trans = pctransform(ptc,tform);
    %pcshow(ptc_trans, VerticalAxis="Y", VerticalAxisDir="Up", ViewPlane="YX");
    
    ptc_trans_resh=reshape(ptc_trans.Location,[],3);
    
    cam1intr=stereoParams.CameraParameters2.K;
    focalLength    = [cam1intr(1,1) cam1intr(1,1)]; 
    principalPoint = [cam1intr(1,3) cam1intr(2,3)];
    imsize=stereoParams.CameraParameters1.ImageSize;
    imageSize      = [imsize(1) imsize(2)];
    intrinsics = cameraIntrinsics(focalLength,principalPoint,imageSize);
    
    tform=rigidtform3d([1 0 0 0;0 1 0 0;0 0 1 0;0 0 0 1]);
    %worldpts=reshape(ptc_trans.Location,[size(ptc_trans.Location,1)*size(ptc_trans.Location,2),3]);
    imgpts=world2img(ptc_trans_resh,tform,intrinsics);
    imgpts=round(imgpts);
    
    x = ptc_trans_resh(:,1);
    y = ptc_trans_resh(:,2);
    z = ptc_trans_resh(:,3);
    s=(x.^2+y.^2+z.^2);
    depth=s.^0.5;
    depth=uint16(depth);
    depth_img=zeros(imsize);
    
    for i=1:size(imgpts,1)
        y=imgpts(i,1);
        x=imgpts(i,2);
        if x>1 && x<=imsize(1) && y>1 && y<=imsize(2)
            if depth_img(x,y)==0
                depth_img(x,y)=depth(i);
            elseif depth(i) < depth_img(x,y)
                depth_img(x,y)=depth(i);
            end
        end
    end
    
    %remove assen balana points
    
    %load the related rgb image
    [L,N] = superpixels(mobileImg,500);
    BW = boundarymask(L);
    idx = label2idx(L);
    
    for labelVal =1:N
        regionIdx = idx{labelVal};
        depth_region=depth_img(regionIdx);
        %fileter out very large values
        depth_region_selected=depth_region(depth_region>0);
        range=max(depth_region_selected,[],'all') - min(depth_region_selected,[],'all');
        m=mean(depth_region_selected);
        if range > 500
            where=find(depth_region>m);
            depth_img(regionIdx(where))=0;
        end
    end
    %imshow(depth_img/max(depth_img,[],'all'));
    final=fill_depth_values(depth_img);