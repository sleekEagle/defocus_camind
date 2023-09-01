import cv2
import os

datapath='D:\\data\\nyu_depth_v2\\resized\\f_40_resized_0.8\\refocused_f_40_fdist_2\\'
output_path='D:\\data\\nyu_depth_v2\\resized\\f_40_resized_0.8\\padded\\refocused_f_40_fdist_2\\'
isdepth=False
#minimum size of the final image
size=(480,480)

#pad images to size
def pad():
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    dir_list = os.listdir(datapath)
    image_files=[file for file in dir_list if file.split('.')[-1]=='png' or file.split('.')[-1]=='jpg']
    for file in image_files:
        if isdepth:
            image = cv2.imread(os.path.join(datapath,file),cv2.IMREAD_UNCHANGED)
            h,w,=image.shape
        else:
            image = cv2.imread(os.path.join(datapath,file))
            h,w,_=image.shape
        border_w=max(0,size[0]-image.shape[0])
        border_h=max(0,size[1]-image.shape[1])
        if(border_h>0 or border_w>0):
            padded=cv2.copyMakeBorder(image, border_w, 0, 0, border_h, cv2.BORDER_CONSTANT, None, value = 0)
        cv2.imwrite(os.path.join(output_path,file),padded)

scale_factor=0.8
def resize():
    output_dir=os.path.join(output_path,str(scale_factor))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dir_list = os.listdir(datapath)
    image_files=[file for file in dir_list if file.split('.')[-1]=='png' or file.split('.')[-1]=='jpg']
    for file in image_files:
        if isdepth:
            image = cv2.imread(os.path.join(datapath,file),cv2.IMREAD_UNCHANGED)
            h,w,=image.shape
        else:
            image = cv2.imread(os.path.join(datapath,file))
            h,w,_=image.shape

        resized_dim=(int(w*scale_factor),int(h*scale_factor))

        #resize operation
        resized = cv2.resize(image, (resized_dim), interpolation = cv2.INTER_AREA)
        if pad:
            border_w=max(0,size[0]-resized.shape[0])
            border_h=max(0,size[1]-resized.shape[1])
            if(border_h>0 or border_w>0):
                resized=cv2.copyMakeBorder(resized, border_w, 0, 0, border_h, cv2.BORDER_CONSTANT, None, value = 0)

        #save image
        cv2.imwrite(os.path.join(output_dir,file),resized)

        # cv2.imshow("Resized image", resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



