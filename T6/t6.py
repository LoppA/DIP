# Lucas de Oliveira Pacheco 9293182 SCC0251 2018/1 Trabalho 6 - Restauracao de Imagens

import imageio
import numpy as np
import math

try:
    import matplotlib.pyplot as plt
except Error:
    pass

def plot(img):
    try:
        plt.imshow(img, cmap='gray')
        plt.show()
    except Error:
        pass

def wrap(img, n):
    m = n//2
    ret = np.zeros((img.shape[0] + 2*m, img.shape[1] + 2*m))

    for i in range(img.shape[0] + 2*m):
        for j in range(img.shape[1] + 2*m):
            ii = (i - m)%img.shape[0]
            jj = (j - m)%img.shape[1]
            ret[i, j] = img[ii, jj]

    return ret

def filtro_adaptativo_reducao(img, n, alpha, EPS = 0.001):
#    img = np.zeros((3,3))
#    for i in range(3):
#        for j in range(3):
#            img[i,j] = i + j
#    print(img)
#    img = wrap(img, 3)
#    print(img)

#    print(np.mean(img[3-1:3+1+1,3-1:3+1+1])) 
#    print(np.var(img[3-1:3+1+1,3-1:3+1+1])) 
#    print(img[3-1:3+1+1,3-1:3+1+1]) 

    ret = np.zeros(img.shape).astype(np.float)
    img = wrap(img, n).astype(np.float)

    var = alpha*alpha
    m = n//2

    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            ii = i + m
            jj = j + m

            filtro = img[ii-m:ii+m+1, jj-m:jj+m+1]

            media = np.mean(filtro)
            var_local = np.var(filtro)

            assert(ii-m >= 0)
            assert(jj-m >= 0)
            assert(ii + m + 1 <= img.shape[0])
            assert(jj + m + 1 <= img.shape[1])

            if(var_local < EPS):
                ret[i,j] = img[ii,jj]
            else:
                ret[i,j] = img[ii,jj] - (var*(img[ii,jj] - media))/var_local

    return ret.astype(np.uint8)


Icomp = imageio.imread(str(input()).rstrip())
Inoisy = imageio.imread(str(input()).rstrip())
op = int(input())
N = int(input())

plot(Inoisy)

if op == 1:
    alpha = float(input())
    Iout = filtro_adaptativo_reducao(Inoisy, N, alpha)
elif op == 2:
    print("Todo", 2)
else:
    print("Todo", 3)

assert(Iout.shape == Icomp.shape)

plot(Iout)

R, S  = Iout.shape
rmse = np.sqrt(np.sum(np.square(Icomp - Iout)) / (R * S))
print("%.5f" % rmse)
