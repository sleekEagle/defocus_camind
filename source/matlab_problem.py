import numpy as np
import cv2


# Create a black image
h=w=512
color = (0,0,0)
img = np.ones((h,w,3), np.uint8)*255
# img[:,:,0]*=40
# img[:,:,1]*=60
# img[:,:,2]*=99
cv2.circle(img,(int(h/2),int(w/2)), 40,color,-1)
cv2.imwrite(r'C:\Users\lahir\data\matlabtest\testimage\img.png',img)

depth = np.ones((h,w), np.uint16)*2000
cv2.circle(depth,(int(h/2),int(w/2)), 40,1000,-1)
cv2.imwrite(r'C:\Users\lahir\data\matlabtest\depth\img.png',depth)



cv2.imshow('I2',img)
cv2.waitKey(0)
cv2.destroyAllWindows()