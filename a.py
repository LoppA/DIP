import numpy as np  # arrays multidimensionais
import imageio      # carregar e salvar imagens
import sys          # funcoes do sistema
import matplotlib.pyplot as plt # exibicao de imagens, graficos, etc


# abrir imagens
filename1 = 'images/numeros1.png'
filename2 = 'images/numeros2.png'

img1 = imageio.imread(filename1)
img2 = imageio.imread(filename2)

# checar se tem o mesmo tamanho
assert img1.shape == img2.shape

# imgout eh a diferenca entre as imagem, cast para evita overflow
imgout = img1.astype(np.float) - img2.astype(np.float)

# valor minimo e maximo de imgout
imin = np.min(imgout)
imax = np.max(imgout)

# normalizando imgout para 0-255 e castando para inteiro de 8 bits
imgout_norm = (imgout-imin)/(imax-imin)
imgout_norm = (imgout_norm*255).astype(np.uint8)

# plotando
plt.imshow(imgout_norm, cmap='gray')
plt.colorbar()
plt.show()

# salvando
imageio.imwrite("images/numeros_diff.jpg", imgout_norm)
