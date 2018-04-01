# Lucas de Oliveira Pacheco 9293182
# SCC0251   Trabalho 2 - realce e superresolucao

import numpy as np
import imageio
from math import sqrt

# Combinar imagens L1, L2, L3 e L4 em uma imagem H de maior resolucao
# dobro de largura e dobro de altura
def superres(L1, L2, L3, L4):
    assert (L1.shape == L2.shape == L3.shape == L4.shape)

    dimL = L1.shape
    dimH = (dimL[0]*2, dimL[1]*2)
    H = np.zeros(dimH, dtype=np.uint8)

    # Adiciona os pixels de L1 em H
    for y in range(dimL[0]):
        for x in range(dimL[1]):
            H[2*y, 2*x] = L1[y,x]

    # Adiciona os pixels de L2 em H
    for y in range(dimL[0]):
        for x in range(dimL[1]):
            H[2*y + 1, 2*x] = L2[y,x]

    # Adiciona os pixels de L3 em H
    for y in range(dimL[0]):
        for x in range(dimL[1]):
            H[2*y, 2*x + 1] = L3[y,x]

    # Adiciona os pixels de L4 em H
    for y in range(dimL[0]):
        for x in range(dimL[1]):
            H[2*y + 1, 2*x + 1] = L4[y,x]

    return H

# Equalizacao de histogramas
def hist(img):
    h = np.zeros(256, dtype=int)

    # constroi histograma
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            h[img[y,x]] += 1

    # histograma acumulado
    for i in range(1, 256):
        h[i] += h[i-1]

    N, M = img.shape

    # Equalizacao
    ret = np.zeros(img.shape, dtype=np.uint8)
    for r in range(256):
        new_r = (h[r]*255.0)/float(M*N)
        ret[np.where(img == r)] = new_r.astype(np.uint8)

    return ret

# Ajuste gamma
def ajuste_gamma(img, gamma):
    c = 255.0
    img = (c * np.power(img.astype(np.float)/c, 1.0/gamma))

    return img.astype(np.uint8)


# Leitura de parametros
imglow = str(input()).rstrip()
imghigh = str(input()).rstrip()
metodo = int(input())
gamma = float(input())

# Leitura das imagens
L1 = imageio.imread(imglow + '1.png')
L2 = imageio.imread(imglow + '2.png')
L3 = imageio.imread(imglow + '3.png')
L4 = imageio.imread(imglow + '4.png')

if metodo == 1: # Histograma individual
    L1 = hist(L1)
    L2 = hist(L2)
    L3 = hist(L3)
    L4 = hist(L4)
elif metodo == 3:   # Ajuste Gamma
    L1 = ajuste_gamma(L1, gamma)
    L2 = ajuste_gamma(L2, gamma)
    L3 = ajuste_gamma(L3, gamma)
    L4 = ajuste_gamma(L4, gamma)

H = superres(L1, L2, L3, L4)

if metodo == 2: # Histograma das imagens L1+L2+L3+L4
    H = hist(H)

# Calculo do erro
REF = imageio.imread(imghigh + '.png')

assert (H.shape == REF.shape)

rmse = sqrt((1.0/(H.shape[0]*H.shape[1]))* np.sum(np.square(REF.astype(np.float) - H.astype(np.float))))

print("%.4f" % rmse)
