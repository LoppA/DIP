# Lucas de Oliveira Pacheco 9293182 SCC0251 2018/1 Trabalho 5 - inpainting usando FFTs 

import numpy as np
import imageio
from math import sqrt

#import matplotlib.pyplot as plt

# Filtra Gk, zerando coeficientes das frequencias de Gk relativas a:
# maiores ou iguais a 90% do maximo da magitude de M
# menores ou iguias a 1% do maximo da magnitude de Gk
def filtra(Gk, M):
    Gk[np.where(np.logical_or(Gk >= 0.9 * M, Gk <= 0.01 * Gk))] = 0
    return Gk

# Realiza convolucao de img com um filtro de media 'sz' x 'sz' no dominio da frequencia
def conv(img, sz):
    filt = np.zeros(img.shape)
    filt[0:sz, 0:sz] = np.ones([sz, sz]) / (sz*sz)
    filt = np.fft.fft2(filt)

    return np.multiply(filt,img)

# normaliza a imagem em valores entra 0 e 255
def norm(img):
    img = np.real(img)
    mn = np.min(img)
    mx = np.max(img)

    img = (img-mn)/(mx-mn)
    img = (img * 255)

    return img.astype(np.uint8)

# insere os pixels conhecidos na estimativa
def insert(gk, g0, mask):
    return (1-(mask/255))*g0 + (mask/255)*gk

# funcao que aplica o algoritmo de Gerchberg-Papoulis em uma
# imagem g, com mascara mask, iterando T vezes
def gerchberg_papoulis(g, mask, T):
    gk = g
    M = np.fft.fft2(mask)   # transformada de Fourier da mascara M

#    plt.imshow(gk, cmap='gray')
#    plt.colorbar()
#    plt.show()
    
    for k in range(T):
        Gk = np.fft.fft2(gk)      # transformada de Fourier da imagem gk

        Gk = filtra(Gk, M)        # filtra Gk

        gk = conv(gk, 7)          # convolucao de gk com filtro de media 7x7

        gk = np.fft.ifft2(Gk)     # transformada inversa

        gk = norm(gk)             # normaliza gk 

        gk = insert(gk, g, mask)  # insere os pixels conhecido

#        plt.imshow(gk, cmap='gray')
#        plt.colorbar()
#        plt.show()

    return gk

imgo = imageio.imread(str(input()).rstrip())    # imagem original
imgi = imageio.imread(str(input()).rstrip())    # imagem deteriorada
imgm = imageio.imread(str(input()).rstrip())    # mascara
T = int(input())    # numero de iteracoes

res = gerchberg_papoulis(imgi, imgm, T)

# calculo do rmse
assert (imgo.shape == res.shape)
rmse = sqrt(np.sum(np.square(imgo-res)) / (imgo.shape[0]*imgo.shape[1]))
print("%.5f" % rmse)
