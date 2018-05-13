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

def filtro_adaptativo_reducao(img, n, alpha, EPS = 0.001):
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

def filtro_adaptativo_mediana(img, n, M):
    ret = np.zeros(img.shape).astype(np.float)
    img = wrap(img, M).astype(np.float)

    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            ii = i + M//2
            jj = j + M//2

            while(n <= M):
                m = n//2

                if(n%2 == 1):
                    filtro = img[ii-m:ii+m+1, jj-m:jj+m+1]
                else:
                    filtro = img[ii-m+1:ii+m+1, jj-m+1:jj+m+1]

                zmed = np.median(filtro)
                zmin = np.amin(filtro)
                zmax = np.amax(filtro)

                a1 = zmed - zmin
                a2 = zmed - zmax

                if(a1 > 0 and a2 < 0):
                    b1 = img[ii, jj] - zmin
                    b2 = zmed - zmax
                    if(b1 > 0 and b2 < 0):
                        ret[i,j] = img[ii,jj]
                    else:
                        ret[i,j] = zmed
                    break
                else:
                    n+=1
                    if(n > M):
                        ret[i,j] = zmed
    

    return ret.astype(np.uint8)


Icomp = imageio.imread(str(input()).rstrip())
Inoisy = imageio.imread(str(input()).rstrip())
op = int(input())
N = int(input())

if op == 1:
    alpha = float(input())
    Iout = filtro_adaptativo_reducao(Inoisy, N, alpha)
elif op == 2:
    M = int(input())
    Iout = filtro_adaptativo_mediana(Inoisy, N, M)
else:
    print("Todo", 3)

assert(Iout.shape == Icomp.shape)
'''
import matplotlib.pyplot as plt

plt.imshow(Inoisy, cmap='gray')
plt.show()

plt.imshow(Iout, cmap='gray')
plt.show()
'''
R, S  = Iout.shape
rmse = np.sqrt(np.sum(np.square(Icomp - Iout)) / (R * S))
print("%.5f" % rmse)
