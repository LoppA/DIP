# Lucas de Oliveira Pacheco 9293182 SCC0251 2018/1 Trabalho 6 - Restauracao de Imagens

import imageio
import numpy as np

def wrap(img, n):
    m = n//2
    ret = np.zeros((img.shape[0] + 2*m, img.shape[1] + 2*m))

    for i in range(img.shape[0] + 2*m):
        for j in range(img.shape[1] + 2*m):
            ii = (i - m)%img.shape[0]
            jj = (j - m)%img.shape[1]
            ret[i, j] = img[ii, jj]

    return ret

def filtro_adaptativo_reducao(img, n, alpha):
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
    img = wrap(img, n)

    var = alpha*alpha

    return ret.astype(np.uint8)


Icomp = imageio.imread(str(input()).rstrip())
Inoisy = imageio.imread(str(input()).rstrip())
op = int(input())
N = int(input())

if op == 1:
    alpha = float(input())
    Iout = filtro_adaptativo_reducao(Inoisy, N, alpha)
elif op == 2:
    print("Todo", 2)
else:
    print("Todo", 3)

assert(Iout.shape == Icomp.shape)

R, S, _  = Iout.shape
rmse = np.sqrt(np.sum(np.square(Icomp - Iout)) / (R * S))
