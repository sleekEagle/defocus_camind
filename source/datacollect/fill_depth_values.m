function final = fill_depth_values(depth_img)
    binary_depth=uint8(depth_img>0);
    zero_values=binary_depth==0;
    
    filt_vert=[0 1 0;0 0 0;0 1 0];
    filt_hori=[0 0 0;1 0 1;0 0 0];
    vert_img=imfilter(binary_depth,filt_vert);
    vert_img=vert_img==2;
    
    hori_img=imfilter(binary_depth,filt_hori);
    hori_img=hori_img==2;
        
    to_hori_fill=logical(hori_img.*zero_values);
    to_vert_fill=logical(vert_img.*zero_values);

    hori_interp=[0 0 0;0.5 0 0.5;0 0 0];
    vert_interp=[0 0.5 0;0 0 0;0 0.5 0];
    
    hori_filled=imfilter(depth_img,hori_interp);
    hori_filled=hori_filled.*to_hori_fill;
    
    vert_filled=imfilter(depth_img,vert_interp);
    vert_filled=vert_filled.*to_vert_fill;
    
    final=depth_img+hori_filled+vert_filled;
