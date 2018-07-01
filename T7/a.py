# Lucas de Oliveira Pacheco 9293182 SCC0251 2018/1 Trabalho 7 - Descrição de imagens

import imageio
import numpy as np

def gray(img):
    gray_img = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gray_img[i, j] = int(0.299*img[i,j,0] + 0.587*img[i,j,1] + 0.114*img[i,j,2])

    return gray_img

def quant(img, b):
    img = img>>(8-b)
    return img

def hist_cor(img, b):
    hist = np.zeros((1<<b)).astype(np.float64)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i,j]] += 1

    img_size = img.shape[0] * img.shape[1]

    hist /= img_size

    return hist

def haralick(img, Q, tons):
    A = np.zeros((tons, tons)).astype(np.float64)

    for i in range(img.shape[0] - 1):
        for j in range(img.shape[1] - 1):
            A[img[i,j]][img[Q(i,j)]] += 1

    sum = (img.shape[0] - 1)*(img.shape[1] - 1)
    assert(sum > 0)

    A /= sum

    return A

def energia(G):
    val = 0
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            val += G[i,j]*G[i,j]
    return val

def entropia(G):
    val = 0
    EPS = 0.001
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            val += G[i,j] * np.log(G[i,j] + EPS)
    return -val

def contraste(G):
    val = 0
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            val += (i-j)*(i-j)*G[i,j]
    return val/((G.shape[0]-1) * (G.shape[1]-1))

def correlacao(G):
    ui = 0
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            ui += i*G[i,j]
    uj = 0
    for j in range(G.shape[1]):
        for i in range(G.shape[0]):
            uj += j*G[i,j]

    oi = 0
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            oi += (i-ui)*(i-ui)*G[i,j]

    oj = 0
    for j in range(G.shape[1]):
        for i in range(G.shape[0]):
            oj += (j-uj)*(j-uj)*G[i,j]

    if oi*oj <= 0:
        return 0

    val = 0
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            val += i*j*G[i,j] - ui*uj

    return val/(oi*oj)

def homogeneidade(G):
    val = 0
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            val += G[i,j]/(1 + abs(i-j))
    return val

def descritores(G):
    desc = np.zeros(5).astype(np.float64)
    return desc

    desc[0] = energia(G)
    desc[1] = entropia(G)
    desc[2] = contraste(G)
    desc[3] = correlacao(G)
    desc[4] = homogeneidade(G)

    return desc

def to_graus(x):
    return (180.0 * x)/np.pi

def gradiente(G):
    hist = np.zeros(18).astype(np.float64)

    wx = np.zeros((3,3))
    wx[0] = [-1, 0, 1]
    wx[1] = [-2, 0, 2]
    wx[2] = [-1, 0, 1]

    wy = np.zeros((3,3))
    wy[0] = [-1, -2, -1]
    wy[1] = [0, 0, 0]
    wy[2] = [1, 2, 1]

    sum = 0
    for i in range(G.shape[0]-2):
        for j in range(G.shape[1]-2):
            gx = np.sum(wx*G[i:i+3, j:j+3])
            gy = np.sum(wy*G[i:i+3, j:j+3])
            mag = np.sqrt(gx*gx + gy*gy)
            ang = (np.arctan2(gy, gx) * 180.0 / np.pi)+180.0
            if(ang >= 360.0):
                ang -= 360.0
            hist[int(ang/20)] += mag
            sum += mag

    if(sum > 0): 
        hist /= sum
    return hist

def mapeia(img, b):
    vhist = hist_cor(img, b)

    def deslocamento(x, y):
        return (x+1, y+1)
    mat_haralick = haralick(img, deslocamento, (1<<b))

    vtextura = descritores(mat_haralick)

    vgradiente = gradiente(img)

    return np.concatenate((vhist, vtextura, vgradiente))

def dist(vector_a, vector_b, b):
    assert(vector_a.shape == vector_b.shape)

    val = 0
    tons_cinza = (1<<b)
    for i in range(0, tons_cinza):
        val += np.square(vector_a[i]-vector_b[i]) / tons_cinza
    for i in range(tons_cinza, tons_cinza+5):
        val += np.square(vector_a[i]-vector_b[i]) / 5.0
    for i in range(tons_cinza+5, tons_cinza+5+18):
        val += np.square(vector_a[i]-vector_b[i]) / 18.0

    return np.sqrt(val)

def compara(img, obj, b):
    descritor_obj = mapeia(obj, b)

    min_dist = -1
    min_id = 1
    id = 1
    for y in range(0, img.shape[0] - obj.shape[0] + 1, 16):
        for x in range(0, img.shape[1] - obj.shape[1] + 1, 16):
            descritor_janela = mapeia(img[y:(y+obj.shape[1]), x:(x+obj.shape[0])], b)
            dist_atual = dist(descritor_obj, descritor_janela, b)
            if(min_dist == -1 or dist_atual < min_dist):
                min_dist = dist_atual
                min_id = id
                print(y, y+obj.shape[1], x,x+obj.shape[0])
            id += 1
    return min_id

objeto = imageio.imread(str(input()).rstrip())   # Imagem com objeto de interesse
imagem = imageio.imread(str(input()).rstrip())  # Imagem para realizar busca
b = int(input())    # parametro de quantização de bits

objeto = gray(objeto)
imagem = gray(imagem)

objeto = quant(objeto, b)
imagem = quant(imagem, b)

#print(compara(imagem, objeto, b))

import matplotlib.pyplot as plt

plt.subplot(231)
plt.imshow(objeto, cmap='gray')
#plt.axis('off')
plt.subplot(232)
plt.imshow(imagem, cmap='gray')
img_1 = imagem[16:48, 0:32]
plt.subplot(234)
plt.imshow(img_1, cmap='gray')
img_2 = imagem[208:240, 112:144]
plt.subplot(235)
plt.imshow(img_2, cmap='gray')
img_3 = imagem[0:32, 16:48]
plt.subplot(236)
plt.imshow(img_3, cmap='gray')
img_4 = imagem[144:176, 112:144]
plt.subplot(233)
plt.imshow(img_4, cmap='gray')
plt.show()

print(compara(imagem, objeto, b))

descritor_obj = mapeia(objeto, b)
descritor_janela = mapeia(img_1, b)
print(dist(descritor_obj, descritor_janela, b))

descritor_obj = mapeia(objeto, b)
descritor_janela = mapeia(img_2, b)
print(dist(descritor_obj, descritor_janela, b))

descritor_obj = mapeia(objeto, b)
descritor_janela = mapeia(img_3, b)
print(dist(descritor_obj, descritor_janela, b))

descritor_obj = mapeia(objeto, b)
descritor_janela = mapeia(img_4, b)
print(dist(descritor_obj, descritor_janela, b))
