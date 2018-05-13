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


def filtro_media_contra_harmonica(img, n, Q):
    ret = np.zeros(img.shape).astype(np.float)

    m = n//2

    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            i1 = max(i-m, 0)
            i2 = min(i+m+1, ret.shape[0])

            j1 = max(j-m, 0)
            j2 = min(j+m+1, ret.shape[1])

            filtro = img[i1:i2, j1:j2]

            if(np.amin(filtro) == 0):
                ret[i,j] = img[i,j]
            else:
                A = np.sum(np.power(filtro, Q + 1))
                B = np.sum(np.power(filtro, Q))

                if(B == 0):
                    ret[i,j] = img[i,j]
                else:
                    ret[i,j] = A/B

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
    Q = float(input())
    Iout = filtro_media_contra_harmonica(Inoisy, N, Q)

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
