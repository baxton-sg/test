

import os
import sys
import csv
import numpy as np
import scipy.misc as spm

from dicom_wr import DICOM
from utils_wr import UTILS
from ann import ANN
from pathes import *
import core
from core import *


utils = UTILS()



IMG_ROWS = 60
IMG_COLS = 60


COLUMNS = 24

MEAN = 0
RANGE = 0



# meta parameters 
STRIDE = 1

# 0.819794341899 on full POSITIONS
EPOCHES, MEAN_MUL, FRAME_SIZE, KOEF, LOW_VAL, HIGH_VAL, CUTOFF = (1, 1.5191511044878414, 3, 2.7982127210771974, 8.9221054386835821, 25.033605786648987, 2.6)









def process(in_dir, out_dir, regressor, case, scan, pos): 

    path = os.path.join(in_dir, case, "study", scan)
    path_out = os.path.join(out_dir, case, "study", scan)
    mkdir_and_clear(path_out)

    fnames = [f for f in os.listdir(path) if "dcm" in f and not f.startswith('.')]
    files_num = len(fnames)
   


    BUFFERS = {}
    METAS   = {}
    ORIGS   = {}
    KEYS    = {}

    #
    # load data
    #
    buffering(path, BUFFERS, ORIGS, METAS, KEYS, CUTOFF)

#    print pos

    if None == pos:

        # 
        # get rectangles
        #
        POSITIONS = {}
        RECTANGLES = {}
        core.process(path, POSITIONS, RECTANGLES, BUFFERS, ORIGS, KEYS, \
                    EPOCHES, MEAN_MUL, FRAME_SIZE, STRIDE, KOEF, LOW_VAL, HIGH_VAL, remove_zeros=False)

        rects = RECTANGLES[path]

        #
        # get the best rectangle
        #
        bestR = None
        bestP = 0
        data = np.zeros((1,COLUMNS), dtype=np.float64)
        for i in range(len(rects)):
            r = rects[i]
    
            data[0,:20] = METAS[path]
            data[0,20:] = r[:4]

            data = norm(data)

            predictions = regressor.predict_proba(data)
            P = predictions[0, 1]

            if P > bestP:
                bestP = P
                bestR = rects[i]

    else:
        bestR = pos
        bestP = 1.

 
    #
    # cut rects
    #
#    import matplotlib.pyplot as plt
#
#    SP = [6, 6, 1]
#    plt.clf()
#    plt.subplot(*SP)
#
#    b = BUFFERS[KEYS[path][0]]
#    plt.imshow(b)
#    SP[-1] += 1
 
    for f in range(files_num):
        key = KEYS[path][f]
        dicom = DICOM()
        dicom.verbose(False)
        dicom.fromfile(key)
        orig = dicom.img_buffer().astype(np.float64)
        dicom.free()
        b = BUFFERS[key]

        print bestR, "(*)"

        if None == pos:
            VK = float(b.shape[0]) / orig.shape[0]
            HK = float(b.shape[1]) / orig.shape[1]
            bestP = (bestR[0] / VK, bestR[1] / VK, bestR[2] / HK, bestR[3] / HK, 0)
        print bestR
        b = orig
        

        print bestR, bestP
        r = b[bestR[0] : bestR[1] + 1, bestR[2] : bestR[3] + 1].copy()

        #b = utils.img_resize(r, IMG_ROWS, IMG_COLS)
        b = r

        fn = os.path.join(path_out, fnames[f]) + ".png"
        with open(fn, "wb+") as fout:
            spm.imsave(fn, b)

#        plt.subplot(*SP)
#        SP[-1] += 1
#        plt.imshow(b)
#    plt.show()

            
            

def norm(data):
    return (data - MEAN) / RANGE








def main(): 
    global MEAN, RANGE

    # "train" or "validate" - names of dirs
    train_validate = sys.argv[1]
    
    path = os.path.join(PATH_DATA, train_validate)

    regressor, MEAN, RANGE = load_ann(os.path.join(PATH_CACHE, "sq_ann.bin"))

    positions, indices = load_positions()

    for case in [d for d in os.listdir(path) if not d.startswith('.')]:
        path_case = os.path.join(path, case, "study")
        for scan in [d for d in os.listdir(path_case) if not d.startswith('.')]:
            print case, scan
            key = os.path.join(case, "study", scan)
            pos = None
            if key in positions:
                pos = positions[key]
            process(path, path + "_h2", regressor, case, scan, pos)





if __name__ == "__main__":
    main()






