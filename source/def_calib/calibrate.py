import cv2
import numpy as np

img=cv2.imread(r'C:\Users\lahir\Downloads\circles.jpg',cv2.IMREAD_GRAYSCALE)

# Setup SimpleBlobDetector parameters.
blobParams = cv2.SimpleBlobDetector_Params()

# Change thresholds
blobParams.minThreshold = 8
blobParams.maxThreshold = 255

# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 200  # minArea may be adjusted to suit for your experiment
blobParams.maxArea = 2500   # maxArea may be adjusted to suit for your experiment

# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.1

# Filter by Convexity
blobParams.filterByConvexity = True
blobParams.minConvexity = 0.87

# Filter by Inertia
blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.01

# Create a detector with the parameters
blobDetector = cv2.SimpleBlobDetector_create(blobParams)

keypoints = blobDetector.detect(img)

im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findCirclesGrid(im_with_keypoints, (6,8), None, flags = cv2.CALIB_CB_ASYMMETRIC_GRID)  

cv2.imshow('name',im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()



cv2.findCirclesGrid(img,(6,8),None,cv2.CALIB_CB_ASYMMETRIC_GRID,blobDetector)




