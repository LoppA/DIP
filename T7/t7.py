# Lucas de Oliveira Pacheco 9293182 SCC0251 2018/1 Trabalho 7 - Descrição de imagens

import imageio
import numpy as np

def gray(img):
    '''
        Converte img para tons de cinza
            img: imagem a ser convertida
            retorno -> gray_img: imagem em tons de cinza
    '''
    gray_img = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gray_img[i, j] = int(0.299*img[i,j,0] + 0.587*img[i,j,1] + 0.114*img[i,j,2])

    return gray_img

def quant(img, b):
    '''
        Quantização da imagem img com b bits
        img: imagem
        b: numero de bits
        retorno -> img: imagem quantizada
    '''
    img = img>>(8-b)
    return img

def hist_cor(img, b):
    hist = np.zeros((1<<b)).astype(np.float64)
    '''
        Retorna histograma normalizado da ocorrencia de tons de cinza na imagem img
        img: imagem a ser processada
        b: quantidade de bits que representa os tons de cinza
        retorno -> hist: histograma normalizado
    '''

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i,j]] += 1

    # normalização
    img_size = img.shape[0] * img.shape[1]
    hist /= img_size

    return hist

def haralick(img, Q, tons):
    '''
        Computa a matriz de haralick sobre a imagem img, com 'tons' tons de cinza e operador de deslocamento Q
        img: imagem a ser processada
        Q: funca de deslocamento
        tons: tons de cinza
        returno -> A : matriz de haralick
    '''
    A = np.zeros((tons, tons)).astype(np.float64)

    for i in range(img.shape[0] - 1):
        for j in range(img.shape[1] - 1):
            A[img[i,j]][img[Q(i,j)]] += 1

    # normalização
    sum = (img.shape[0] - 1)*(img.shape[1] - 1)
    assert(sum > 0)
    A /= sum

    return A

def energia(G):
    '''
        Calcula o descritor de energia sobre uma matriz de haralick G
        G: matriz de haralick
        retorno -> descritor de energia
    '''
    return np.sum(G*G)

def entropia(G):
    '''
        Calcula o descritor de entropia sobre uma matriz de haralick G
        G: matriz de haralick
        retorno -> descritor de entropia
    '''
    EPS = 0.001
    return -np.sum(G * np.log(G + EPS))

def contraste(G):
    '''
        Calcula o descritor de contraste sobre uma matriz de haralick G
        G: matriz de haralick
        retorno -> descritor de contraste
    '''
    ij = np.array([[(i-j)*(i-j) for i in range(G.shape[0])] for j in range(G.shape[1])])
    return (np.sum(ij*G)) / ((G.shape[0]-1)*(G.shape[1]-1))

def correlacao(G):
    '''
        Calcula o descritor de correlacao sobre uma matriz de haralick G
        G: matriz de haralick
        retorno -> descritor de correlacao
    '''
    # Calculo da constante ui
    vi = np.array([[i for j in range(G.shape[1])] for i in range(G.shape[0])])
    ui = np.sum(vi * G)

    # Calculo da constante uj
    vj = np.array([[j for j in range(G.shape[1])] for i in range(G.shape[0])])
    uj = np.sum(vj * G)

    # Calculo da constante oi
    iui = np.array([[(i-ui)*(i-ui) for j in range(G.shape[1])] for i in range(G.shape[0])])
    oi = np.sum(iui * G)

    # Calculo da constante oj
    juj = np.array([[(j-uj)*(j-uj) for j in range(G.shape[1])] for i in range(G.shape[0])])
    oj = np.sum(juj * G)

    if oi*oj <= 0:
        return 0

    # Calculo do descrito de correlacao
    val = np.sum([[i*j*G[i,j] - ui*uj for j in range(G.shape[1])] for i in range(G.shape[0])])
        
    return val/(oi*oj)

def homogeneidade(G):
    '''
        Calcula o descritor de homogeneidade sobre uma matriz de haralick G
        G: matriz de haralick
        retorno -> descritor de homogeneidade
    '''       
    absij = np.array([[(1+abs(i-j)) for i in range(G.shape[0])] for j in range(G.shape[1])])
    return np.sum(G / absij)
    
def descritores(G):
    '''
        Calculo de descritores de haralick: energia, entropia, constraste, correlacao
        e homogeneidade. Sobre uma matriz de haralick G
        G: Matriz de haralick
        retorno -> desc: array numpy de tamanho 5 com os descritores normalizados
    '''
    desc = np.zeros(5).astype(np.float64)

    desc[0] = energia(G)
    desc[1] = entropia(G)
    desc[2] = contraste(G)
    desc[3] = correlacao(G)
    desc[4] = homogeneidade(G)

    # Normalização
    sum = np.sum(desc)
    if(sum != 0):
        desc /= sum

    return desc

def gradiente(G):
    '''
        Computar o gradiente da imagem G nas direções x e y, com um histograma
        para cada intervalo discretizado de angulos(faixa de 20 graus)
        G: imagem
        retorno-> hist: array numpy normalizado de tamanho 18 com a soma das magnitudes para cada intervalo de ângulos 
    '''
    hist = np.zeros(18).astype(np.float64)

    # operador de sober na direção x
    wx = np.zeros((3,3))
    wx[0] = [-1, 0, 1]
    wx[1] = [-2, 0, 2]
    wx[2] = [-1, 0, 1]

    # operador de sober na direção y
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

    # Normalização
    if(sum > 0): 
        hist /= sum
    return hist

def mapeia(img, b):
    '''
        Computar um vetor de caracteristicas da imagem
        img: imagem
        b: bits quantizados da imagem
        retorno -> vetor numpy das caracteristicas da imagem
    '''
    # histograma de cor
    vhist = hist_cor(img, b)

    # descritores de textura de haralick
    def deslocamento(x, y):
        return (x+1, y+1)
    mat_haralick = haralick(img, deslocamento, (1<<b))
    vtextura = descritores(mat_haralick)

    # descritores de orientação de gradiente
    vgradiente = gradiente(img)

    return np.concatenate((vhist, vtextura, vgradiente))

def dist(vector_a, vector_b, b):
    '''
        Calcula a distancia de Minkowski com p = 2 entre 2 vetores de caracteristicas
        vector_a: primeiro vetor de caracteristicas
        vector_b: segundo vetor de caracteristicas
        returno : distancia de Minkowski com p = 2 entre os 2 vetores
    '''
    assert(vector_a.shape == vector_b.shape)

    val = 0
    tons_cinza = (1<<b)
    # histograma de cor
    for i in range(0, tons_cinza):
        val += np.square(vector_a[i]-vector_b[i]) / tons_cinza
    # descritores de textura
    for i in range(tons_cinza, tons_cinza+5):
        val += np.square(vector_a[i]-vector_b[i]) / 5.0
    # descritores de orientação de gradiente
    for i in range(tons_cinza+5, tons_cinza+5+18):
        val += np.square(vector_a[i]-vector_b[i]) / 18.0

    return np.sqrt(val)

def compara(img, obj, b):
    '''
        Compara um objeto obj com janelas deslizantes na imagem img com passo 16, e retorna o indice
        da janela com menor vetor de caracteristicas menos distante do vetor de obj
        img: imagem
        obj: objeto
        b: numero de bits quantizados
        retorno-> id: indice da janela com menor disntacia
    '''
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
            id += 1
    return min_id

objeto = imageio.imread(str(input()).rstrip())   # Imagem com objeto de interesse
imagem = imageio.imread(str(input()).rstrip())  # Imagem para realizar busca
b = int(input())    # parametro de quantização de bits

# converte para tons de cinza
objeto = gray(objeto)
imagem = gray(imagem)

# quantiza com b bits
objeto = quant(objeto, b)
imagem = quant(imagem, b)

print(compara(imagem, objeto, b))
