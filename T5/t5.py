# Lucas de Oliveira Pacheco 9293182 SCC0251 2018/1 Trabalho 5 - inpainting usando FFTs 

import numpy as np
import imageio
from math import sqrt

def filtra(Gk, M):
    return Gk

def conv(img, n, m):
    return img

def norm(img):
    mn = np.min(img)
    mx = np.max(img)

    img = (img-mn)/(mx-mn)
    img = (img * 255)

    return img

def insert(g0, gk):
    return gk

def gerchberg_papoulis(g, mask, T):
    gant = g
    M = np.fft.fft2(mask)

    for k in range(1, T + 1):
        Gat = np.fft.fft(gant)      # a done
        Gat = filtra(Gat, M)        # b
        gat = np.fft.ifft2(Gat)     # c done
        gat = conv(gat, 7, 7)       # d
        gat = norm(gat)             # e 
        gat = insert(g, gat)        # f

        gant = gat.astype(np.uint8)

    return gant

imgo = imageio.imread(str(input()).rstrip())    # imagem original
imgi = imageio.imread(str(input()).rstrip())    # imagem deteriorada
imgm = imageio.imread(str(input()).rstrip())    # mascara
T = int(input())

res = gerchberg_papoulis(imgi, imgm, T)

rmse = sqrt(np.sum(imgo-res)**2)
print("%.5f" % rmse)
