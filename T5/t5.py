# Lucas de Oliveira Pacheco 9293182 SCC0251 2018/1 Trabalho 5 - inpainting usando FFTs 

import numpy as np
import imageio
from math import sqrt

import matplotlib.pyplot as plt

# Filtra Gk, zerando coeficientes das frequencias de Gk relativas a:
# maiores ou iguais a 90% do maximo da magitude de M
# menores ou iguias a 1% do maximo da magnitude de Gk
def filtra(Gk, M):
    Gk[np.where(np.logical_or(Gk >= 0.9 * M, Gk <= 0.01 * Gk))] = 0
    return Gk

# Realiza convolucao de img com um filtro de media 'sz' x 'sz' no dominio da frequencia
def conv(img, sz):
#    filt = np.ones([sz, sz]) / (sz*sz)
#    filt = np.fft.fft2(filt)

    filt = np.zeros(img.shape)
    filt[0:7, 0:7] = np.ones([7, 7]) / (49.0)
    filt = np.fft.fft2(filt)

    return filt*img

def norm(img):
    img = np.real(img)
    mn = np.min(img)
    mx = np.max(img)

    img = (img-mn)/(mx-mn)
    img = (img * 255)

    return img.astype(np.uint8)

def insert(gk, g0, mask):
    return (1-(mask/255))*g0 + (mask/255)*gk

def gerchberg_papoulis(g, mask, T):
    gat = g
    M = np.fft.fft2(mask)

    plt.imshow(gat, cmap='gray')
    plt.colorbar()
    plt.show()
    
    for k in range(T):
        Gat = np.fft.fft2(gat)      # a

        Gat = filtra(Gat, M)        # b

        gat = conv(gat, 7)          # d

        gat = np.fft.ifft2(Gat)     # c

        gat = norm(gat)             # e

        gat = insert(gat, g, mask)  # f 

        plt.imshow(gat, cmap='gray')
        plt.colorbar()
        plt.show()

    return gat

imgo = imageio.imread(str(input()).rstrip())    # imagem original
imgi = imageio.imread(str(input()).rstrip())    # imagem deteriorada
imgm = imageio.imread(str(input()).rstrip())    # mascara
T = int(input())    # numero de iteracoes

res = gerchberg_papoulis(imgi, imgm, T)

# calculo do rmse
assert (imgo.shape == res.shape)
rmse = sqrt(np.sum(np.square(imgo-res)) / (imgo.shape[0]*imgo.shape[1]))
print("%.5f" % rmse)
