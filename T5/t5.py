# Lucas de Oliveira Pacheco 9293182 SCC0251 2018/1 Trabalho 5 - inpainting usando FFTs 

import numpy as np
import imageio

imgo = imageio.imread(str(input()).rstrip())    # imagem original
imgi = imageio.imread(str(input()).rstrip())    # imagem deteriorada
imgm = imageio.imread(str(input()).rstrip())    # mascara
T = int(input())

gant = imgi
M = np.fft.fft2(imgm)

# for k in range(1, T + 1):
