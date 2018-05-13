# Lucas de Oliveira Pacheco 9293182 SCC0251 2018/1 Trabalho 6 - Restauracao de Imagens

import imageio
import numpy as np

# Faz uma matriz circular em relacao a um filtro de tamanho n
def wrap(img, n):
    m = n//2
    ret = np.zeros((img.shape[0] + 2*m, img.shape[1] + 2*m))

    for i in range(img.shape[0] + 2*m):
        for j in range(img.shape[1] + 2*m):
            ii = (i - m)%img.shape[0]
            jj = (j - m)%img.shape[1]
            ret[i, j] = img[ii, jj]

    return ret

# Implementacao do Filtro Adaptativo de Reducao do Ruido Local
# Iout = Inoisy - (var_noisy/var_N)*(Inoisy - MeanN)
def filtro_adaptativo_reducao(img, n, alpha, EPS = 0.001):
    ret = np.zeros(img.shape).astype(np.float)
    img = wrap(img, n).astype(np.float)

    var = alpha*alpha
    m = n//2

    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            # pixel de img estao deslocados em relacao a sua posicao original devido ao wrap
            ii = i + m
            jj = j + m

            filtro = img[ii-m:ii+m+1, jj-m:jj+m+1]

            # Calculo da media
            media = np.mean(filtro)
            # Calculo da variacao local
            var_local = np.var(filtro)

            # Se var_local muito pequena mantem pixel
            if(var_local < EPS):
                ret[i,j] = img[ii,jj]
            else:
                # aplicacao da formula
                ret[i,j] = img[ii,jj] - (var*(img[ii,jj] - media))/var_local

    return ret.astype(np.uint8)

# Implementacao do Filtro Adaptativo de Mediana
def filtro_adaptativo_mediana(img, n, M):
    ret = np.zeros(img.shape).astype(np.float)
    img = wrap(img, M).astype(np.float)

    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            # pixel de img estao deslocados em relacao a sua posicao original devido ao wrap
            ii = i + M//2
            jj = j + M//2

            # Etapa A
            while(n <= M):
                m = n//2

                if(n%2 == 1):
                    # se filtro impar deslocamento normal
                    filtro = img[ii-m:ii+m+1, jj-m:jj+m+1]
                else:
                    # se filtro par deslocamento normal o centro é o pixel da esquerda dentre os 2 empatados
                    filtro = img[ii-m+1:ii+m+1, jj-m+1:jj+m+1]

                zmed = np.median(filtro)    # calculo da mediana
                zmin = np.amin(filtro)      # calculo do minimo
                zmax = np.amax(filtro)      # calculo do maximo

                a1 = zmed - zmin
                a2 = zmed - zmax

                if(a1 > 0 and a2 < 0):
                    # Etapa B
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

# Implementacao do Filtro Adaptativo de Reducao do Ruido Local
# Iout = sum(g**(Q+1)) / sum(g**Q)
# g eh a regiao de Inoisy delimitada pelo filtro centrado em x,y
def filtro_media_contra_harmonica(img, n, Q):
    ret = np.zeros(img.shape).astype(np.float)

    m = n//2

    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            # preenchimento pelo valor zero em todas as posicoes fora da imagem
            i1 = max(i-m, 0)
            i2 = min(i+m+1, ret.shape[0])

            j1 = max(j-m, 0)
            j2 = min(j+m+1, ret.shape[1])

            filtro = img[i1:i2, j1:j2]

            # evitar divisao por zero
            if(np.amin(filtro) == 0 and Q < 0.0):
                ret[i,j] = img[i,j]
            else:
                # calculo da formula
                A = np.sum(np.power(filtro, Q + 1))
                B = np.sum(np.power(filtro, Q))

                ret[i,j] = A/B

    return ret.astype(np.uint8)


Icomp = imageio.imread(str(input()).rstrip())   # Imagem original para comparacao
Inoisy = imageio.imread(str(input()).rstrip())  # Imagem ruidosa para filtragem
op = int(input())   # metodo escolhido
N = int(input())    # tamanho do filtro(N): NxN

if op == 1:
    # metodo 1
    alpha = float(input())
    Iout = filtro_adaptativo_reducao(Inoisy, N, alpha)
elif op == 2:
    # metodo 2
    M = int(input())
    Iout = filtro_adaptativo_mediana(Inoisy, N, M)
else:
    # metodo 3
    Q = float(input())
    Iout = filtro_media_contra_harmonica(Inoisy, N, Q)

# calculo e print do erro rmse
assert(Iout.shape == Icomp.shape)
R, S  = Iout.shape
rmse = np.sqrt(np.sum(np.square(Icomp - Iout)) / (R * S))
print("%.5f" % rmse)
