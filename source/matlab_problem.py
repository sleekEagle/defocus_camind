import numpy as np
import cv2


# Create a black image
h=w=512
img = np.ones((h,w), np.uint8)*255
cv2.circle(img,(int(h/2),int(w/2)), 40,0,-1)
cv2.imwrite(r'C:\Users\lahir\data\matlabtest\testimage\img.png',img)

depth = np.ones((h,w), np.uint16)*2.0
cv2.circle(depth,(int(h/2),int(w/2)), 40,1,-1)
cv2.imwrite(r'C:\Users\lahir\data\matlabtest\depth\img.png',depth)



cv2.imshow('I2',img)
cv2.waitKey(0)
cv2.destroyAllWindows()