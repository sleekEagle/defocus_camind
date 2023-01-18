import numpy as np
import matplotlib.pyplot as plt

'''
Calculation is based on 
Single Image Depth Estimation Trained via Depth from Defocus Cues 
by Gur and Wolf 
CVPR 2019
'''
nsamples=10000
s1=np.random.choice([0.1,.15,.3,0.7,1.5], nsamples)
s2=np.random.uniform(low=0.1, high=3.0, size=(nsamples,))
f=2e-3 #focal legth
pixel_size= 5.6e-3#pixel size in mm
'''
the ratio between the length of the output image in pixels and the length of the sensor in pixels
these two can be different from sensorpix due to things such as pixel binning
'''
pixelratio=1

#calculate blur in pixels of the observed image
def getblur_pix(s1,s2,f,N,pixel_size,pixelratio):
    blur=abs(s2-s1)/s2*f**2/(N*(s1-f))*(1/pixel_size)*(pixelratio)
    return blur

#change s1 and s2 and N keep everything else constant
#assume variables are uniformly distributed
N1=1
blur_values1=getblur_pix(s1,s2,f,N1,pixel_size,pixelratio)
N2=5
blur_values2=getblur_pix(s1,s2,f,N2,pixel_size,pixelratio)
N3=8
blur_values3=getblur_pix(s1,s2,f,N3,pixel_size,pixelratio)
N4=10
blur_values4=getblur_pix(s1,s2,f,N4,pixel_size,pixelratio)
N5=20
blur_values5=getblur_pix(s1,s2,f,N5,pixel_size,pixelratio)

plt.hist(blur_values1)
plt.xlabel('blur in pixles')
plt.ylabel('frequency')
plt.title('blur when N=1')
plt.savefig('blur_experiment/N=1.png')
plt.clf()

plt.hist(blur_values4)
plt.xlabel('blur in pixles')
plt.ylabel('frequency')
plt.title('blur when N=10')
plt.savefig('blur_experiment/N=10.png')
plt.clf()

plt.hist(blur_values5)
plt.xlabel('blur in pixles')
plt.ylabel('frequency')
plt.title('blur when N=20')
plt.savefig('blur_experiment/N=20.png')
plt.clf()

'''
Lets multiply blur_pix by camera parameters
'''
norm_blur1=blur_values1*pixel_size*(s1-f)*N1/f**2
norm_blur2=blur_values2*pixel_size*(s1-f)*N2/f**2
norm_blur3=blur_values3*pixel_size*(s1-f)*N3/f**2
norm_blur4=blur_values4*pixel_size*(s1-f)*N4/f**2
norm_blur5=blur_values5*pixel_size*(s1-f)*N5/f**2

plt.hist(norm_blur1)
plt.xlabel('blur in pixles')
plt.ylabel('frequency')
plt.title('normalized blur when N=1')
plt.savefig('blur_experiment/N=1_norm.png')
plt.clf()

plt.hist(norm_blur4)
plt.xlabel('blur in pixles')
plt.ylabel('frequency')
plt.title('normalized blur when N=10')
plt.savefig('blur_experiment/N=10_norm.png')
plt.clf()

plt.hist(norm_blur5)
plt.xlabel('blur in pixles')
plt.ylabel('frequency')
plt.title('normalized blur when N=20')
plt.savefig('blur_experiment/N=20_norm.png')
plt.clf()

plt.show()


'''
Whap happend if we sample N from a uniform distribution also
'''
nsamples=10000
s1=np.random.choice([0.1,.15,.3,0.7,1.5], nsamples)
s2=np.random.uniform(low=0.1, high=3.0, size=(nsamples,))
N=np.random.uniform(low=1, high=20.0, size=(nsamples,))
blur_values=getblur_pix(s1,s2,f,N,pixel_size,pixelratio)

plt.hist(blur_values)
plt.xlabel('blur in pixles')
plt.ylabel('frequency')
plt.title('blur with randomly chosen N')
plt.savefig('blur_experiment/randomN.png')


















