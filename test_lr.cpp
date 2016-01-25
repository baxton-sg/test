

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utils_wr import UTILS

utils = UTILS()





img = mpimg.imread("test.png")
img_big = utils.img_resize(img, img.shape[0], img.shape[1] * 2)



plt.clf()
plt.subplot(131)
plt.imshow(img)
plt.subplot(132)
plt.imshow(img_big)
plt.show()
