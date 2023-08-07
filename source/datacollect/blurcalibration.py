import cv2 as cv2
import matplotlib.pyplot as plt

image = cv2.imread(r"C:\Users\lahir\Downloads\IMG_20230804_161128.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray.shape

values=gray[328:380,300]
plt.plot(values)
plt.show()
cv2.imshow('Grayscale', gray)
cv2.waitKey(0)  
