# Lucas de Oliveira Pacheco 9293182 SCC0251 2018/1 Trabalho 7 - Descrição de imagens

import imageio
import numpy as np

objeto = imageio.imread(str(input()).rstrip())   # Imagem com objeto de interesse
imagem = imageio.imread(str(input()).rstrip())  # Imagem para realizar busca
b = int(input())    # parametro de quantização de bits

