

import os
import numpy as np
import scipy.misc as spm
import matplotlib.pyplot as plt
from pathes import *


def order(case, scan, base_path):
    path = os.path.join(base_path, case, "study", scan)
    fnames = [f for f in os.listdir(path) if not f.startswith('.')]

    indices = range(len(fnames))

    min = 999999
    minidx = 0
    minImg = None

    max = 0
    maxidx = 0
    maxImg = None

    for i, f in enumerate(fnames):
        fn = os.path.join(path, f) 
        img = spm.imread(fn).astype(np.float64)
        m = img.mean()
       
        if m < min:
            min = m
            minidx = i
            minImg = img

        elif m > max:
            max = m
            maxidx = i
            maxImg = img

    print "%s,%s,%s,%s" % (case, scan, fnames[minidx], fnames[maxidx])

#    print minidx, maxidx
#
#    plt.clf()
#    plt.subplot(121)
#    plt.imshow(minImg)
#    plt.subplot(122)
#    plt.imshow(maxImg)
#    plt.show()




base_path = PATH_TRAIN_H2
for case in [d for d in os.listdir(base_path) if not d.startswith('.')]:
    path = os.path.join(base_path, case, "study")
    for scan in [d for d in os.listdir(path) if not d.startswith('.')]:
        order(case, scan, base_path)














