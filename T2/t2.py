# Lucas de Oliveira Pacheco 9293182
# SCC0251   Trabalho 2 - realce e superresolucao

import numpy as np
import imageio
import matplotlib.pyplot as plt
from math import sqrt

def superres (L1, L2, L3, L4) :
    assert (L1.shape == L2.shape == L3.shape == L4.shape)

    dimL = L1.shape
    dimH = (dimL[0]*2, dimL[1]*2)
    H = np.zeros(dimH, dtype=np.uint8)

    for y in range(dimL[0]):
        for x in range(dimL[1]):
            H[2*y, 2*x] = L1[y,x]

    for y in range(dimL[0]):
        for x in range(dimL[1]):
            H[2*y + 1, 2*x] = L2[y,x]

    for y in range(dimL[0]):
        for x in range(dimL[1]):
            H[2*y, 2*x + 1] = L3[y,x]

    for y in range(dimL[0]):
        for x in range(dimL[1]):
            H[2*y + 1, 2*x + 1] = L4[y,x]

    return H

imglow = str(input()).rstrip()
imghigh = str(input()).rstrip()
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

H = superres(L1, L2, L3, L4)

REF = imageio.imread(imghigh + '.png')

assert (H.shape == REF.shape)

plt.subplot(121)
plt.imshow(H, cmap='gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(REF, cmap='gray')
plt.axis('off')
plt.show()

rmse = sqrt((1.0/(H.shape[0]*H.shape[1]))* np.sum(np.square(REF.astype(np.float) - H.astype(np.float))))

print("%.4f" % rmse)

'''
rmse = 0.0
REF = REF.astype(np.float)
H = H.astype(np.float)
for y in range (H.shape[0]):
    for x in range (H.shape[1]):
        rmse += (REF[y,x] - H[y,x])**2
rmse /= (H.shape[0] * H.shape[1])
rmse = sqrt(rmse)

print("%.4f" % rmse)
'''
