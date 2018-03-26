import numpy as np
import imageio
import matplotlib.pyplot as plt

imglow = str(input())
imghigh = str(input())
metodo = int(input())
game = float(input())

L1 = imageio.imread(imglow + '1.png')
L2 = imageio.imread(imglow + '2.png')
L3 = imageio.imread(imglow + '3.png')
L4 = imageio.imread(imglow + '4.png')

plt.subplot(221)
plt.imshow(L1, cmap='gray')
plt.axis('off')
plt.subplot(222)
plt.imshow(L2, cmap='gray')
plt.axis('off')
plt.subplot(223)
plt.imshow(L3, cmap='gray')
plt.axis('off')
plt.subplot(224)
plt.imshow(L4, cmap='gray')
plt.axis('off')
plt.show()
