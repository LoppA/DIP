import numpy as np
import imageio
from math import pi, sqrt

img_name = str(input()).rstrip()
option = int(input())
n = int(input())

w = np.zeros(n)
if option == 1:
    for i in range(n):
        w[i] = float(input())
else:
    gamma = float(input())
    x = np.arange(-int((n-1)/2), int(n/2) + 1)
    w = 1.0/(sqrt(2.0*pi)*gamma) * np.exp(-1*(np.multiply(x,x))/(2.0*gamma*gamma))
    total = np.sum(w)
    w = w/total

dom = int(input())

img = imageio.imread(img_name)

I = np.zeros(img.shape[0] * img.shape[1])
k = 0
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        I[k] = img[i,j]
        k+=1

R = np.zeros(img.shape[0] * img.shape[1])
if dom == 1:
    for i in range(R.shape[0]):
        k = 0
        for j in range(-int((n-1)/2), int(n/2) + 1):
            R[i] += I[(i+j)%I.shape[0]] * w[k]
            k+=1
else:
    print("TO DO")


rmse = sqrt( np.sum(np.power(I - R, 2))/ float(img.shape[0]*img.shape[1]))

print ("%.4f" % rmse)
