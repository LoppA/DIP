# Lucas de Oliveira Pacheco 9293182 SCC0251 2018/1 Trabalho 1 - Gerador de Imagens

import numpy as np
import random
from math import sin, sqrt

# funcao 1, retorna o pixel x,y 
def f1(x, y, q=None):
    return x + y

# funcao 2, retorna o pixel x,y
def f2(x, y, q):
    return abs(sin(x/q) + sin(y/q))

# funcao 3, retorna o pixel x,y
def f3(x, y, q):
    return abs((x/q) - sqrt(y/q))

# funcao 4, retorna o pixel x,y
def f4(x=None, y=None, q=None):
    return random.random()

# funcao 5, retorna a imagem f inteira
def f5(c):
    img = np.zeros((c,c)).astype(np.float)
    
    steps = int(1 + (c*c)/2)
    x = y = 0
    for i in range(0, steps):
        img[x,y] = 1
        # deslocamento x
        dx = random.randint(-1,1) 
        x = (x + dx)%c
        img[x,y] = 1
        # deslocamento y
        dy = random.randint(-1,1) 
        y = (y + dy)%c
        img[x,y] = 1

    return img

# gera a imagem com parametros f, c e q dados
def gen_img (f, c, q) :
    fun = [f1, f2, f3, f4]

    assert(f >= 0 and f < 5)

    # primeiras 4 funcoes sao semelhantes
    if (f < 4):
        img = np.zeros((c,c)).astype(np.float)
        for x in range(0, c):
            for y in range(0, c):
                img[x,y] = fun[f](x, y, q)
        return img

    # quinta funcao gera a imagem inteira sozinha
    return f5(c)

# normaliza imagem img para inteiro de b bits
def norm(img, max_val):
    assert (b <= 16)

    # maximo e minimo
    mn = np.min(img)
    mx = np.max(img)

    # normaliza 0-1
    img = (img-mn)/(mx-mn)

    # passa para 0-max_val
    img = (img*max_val)

    return img

# b bits mais signficativos
def b_bits(img, b):
    return img>>(8-b)

# gera uma matriz dimg com elementos n x n inteiros de 8 bits a partir
# de uma matrix img com c x c elementos
# como n <= c calcula-se a imagem digitalizada utilizando-se operacao
# de maximo local
def dig(img, c, n):
    assert(n <= c)
    dimg = np.zeros((n,n)).astype(np.uint8)

    d = int(c/n)

    for x in range(n):
        for y in range(n):
            dimg[x,y] = np.max(img[x*d:x*d+d,y*d:y*d+d])

    return dimg

# leitura dos parametros
filename = str(input()).rstrip()
c = int(input())
f = int(input()) - 1
q = int(input())
n = int(input())
b = int(input())
s = int(input())

# inicializando semente s
random.seed(s)

# gera f imagem e normaliza entre 0 e (2**16)-1
img_f = gen_img(f, c, q)
img_f = norm(img_f, (1<<16) - 1)

# convertendo f para inteiro de 8 bits com valores entre 0 e 255
img_f = norm(img_f, (1<<8) - 1)
img_f = img_f.astype(np.uint8)

# deslocamento de bits
img_f = b_bits(img_f, b)

# 'digitalizacao' da imagem
img_g = dig(img_f, c, n)

'''
import matplotlib.pyplot as plt

plt.imshow(img_f, cmap='gray')
plt.colorbar()
plt.show()

plt.imshow(img_g, cmap='gray')
plt.colorbar()
plt.show()
'''

##### Comparacao
# carrega a referencia
R = np.load(filename).astype(np.uint8)

import matplotlib.pyplot as plt

ff, axarr = plt.subplots(1,2)

axarr[0].imshow(img_g, cmap='gray')
axarr[1].imshow(R, cmap='gray')
plt.show()

# calcula a funcao
rmse = sqrt(np.sum((img_g - R)**2))

# imprime com 4 casas decimais
print("%.4f" % rmse)
