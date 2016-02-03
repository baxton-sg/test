


import os
import sys
import csv
import numpy as np
import scipy.misc as spm
from pathes import *
from core import *
from utils_wr import UTILS
from ann import ANN

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor


utils = UTILS()



def get_N(base_path):
    N = 0
    for case in [d for d in os.listdir(base_path) if not d.startswith('.')]:
        path = os.path.join(base_path, case, "study")
        for scan in [d for d in os.listdir(path) if not d.startswith('.')]:
            N += 1
    return N



def get_minimax():
    minimax = {}
    with open(os.path.join(PATH_DATA, "minimax.csv"), "r") as fin:   
        reader = csv.reader(fin, delimiter=",")
        for vec in reader:
            ID = int(vec[0])
            scan = vec[1]
            minimax[(ID, scan)] = (vec[2], vec[3])
    return minimax




def get_data(systole, diastole, minimax, base_path, meta, case_scan_id_map):
    keys = minimax.keys()
    N = len(keys)
    to_train = int(N * .8)
    indices = range(N)
    np.random.shuffle(indices)
    indices_train = set(indices[:to_train])


    SHIFT = 10


    F = 4
    S = 2


    ROWS = 300
    COLS = 300

    R = 60
    C = 60


    FLIP_NUM = 1

    META_SIZE = 0
    meta_indices = [4,5,6,7]

    data = np.zeros((to_train * FLIP_NUM, R * C + META_SIZE), dtype=np.float64)
    data_test = np.zeros((N - to_train, R * C + META_SIZE), dtype=np.float64)

    YStrain = np.zeros((to_train * FLIP_NUM,), dtype=np.float64)
    YStest = np.zeros((N - to_train,), dtype=np.float64)
    YDtrain = np.zeros((to_train * FLIP_NUM,), dtype=np.float64)
    YDtest = np.zeros((N - to_train,), dtype=np.float64)

    idx = 0
    idx_test = 0

    frame = np.zeros((ROWS, COLS), dtype=np.float64)


    ttt = 0


    for i in range(N):
        key = keys[i]

        orig_img = spm.imread(os.path.join(base_path, str(key[0]), "study", key[1], minimax[key][0]))
        orig_img = orig_img.astype(np.float64)


        frame.fill(0)
        frame[:orig_img.shape[0], :orig_img.shape[1]] = orig_img
        #img = utils.filter(frame, F, S, 0)
        img = frame
        img = utils.img_resize(img, R, C)


        if i in indices_train:
            if META_SIZE:
                data[idx,:-META_SIZE] = img.flatten()
                data[idx,-META_SIZE:] = meta[case_scan_id_map[key]][meta_indices]
            else:
                if ttt < 10:
                    ttt += 1
                    plt.clf()
                    plt.imshow(img)
                    plt.show()
                data[idx] = img.flatten()
 
            YStrain[idx] = systole[key[0]]
            YDtrain[idx] = diastole[key[0]]
            idx += 1

            if FLIP_NUM > 1:
                if META_SIZE:
                    data[idx,-META_SIZE:] = meta[case_scan_id_map[key]][meta_indices] 
                else:
                    frame.fill(0)
                    frame[:orig_img.shape[0], :orig_img.shape[1]] = orig_img[::-1,:]
                    img = frame
                    img = utils.img_resize(img, R, C)

                    data[idx] = img.flatten()
#                    tmp = np.zeros(img.shape)
#                    D = np.random.randint(4)
#                    if D == 0:   # up
#                        tmp[:-SHIFT,:] = img[SHIFT:,:]
#                    elif D == 1:   # up
#                        tmp[SHIFT:,:] = img[:-SHIFT,:]
#                    elif D == 2:   # up
#                        tmp[:,:-SHIFT] = img[:,SHIFT:]
#                    elif D == 3:   # up
#                        tmp[:,SHIFT:] = img[:,:-SHIFT]
#                    data[idx] = tmp.flatten()


                YStrain[idx] = systole[key[0]]
                YDtrain[idx] = diastole[key[0]]
                idx += 1
        else:
            if META_SIZE:
                data_test[idx_test,-META_SIZE:] = meta[case_scan_id_map[key]][meta_indices]
                data_test[idx_test,:-META_SIZE] = img.flatten()
            else:
                data_test[idx_test] = img.flatten()

            YStest[idx_test] = systole[key[0]]
            YDtest[idx_test] = diastole[key[0]]
            idx_test += 1

    if META_SIZE:
        min = data[:,-META_SIZE:].min(axis=0)
        max = data[:,-META_SIZE:].max(axis=0)
        r = max - min
        r[r==0] = 0.00001
  
        data[:,-META_SIZE:] /= r
        data_test[:,-META_SIZE:] /= r
    
    return data, data_test, YStrain, YStest, YDtrain, YDtest






def main():
    np.random.seed()

    base_path = PATH_TRAIN_H2

    N = get_N(base_path)
    print N

    systole, diastole = load_train_vals()

    minimax = get_minimax()
    #print minimax

    meta, case_scan_id_map = load_metadata()

    data, data_test, YStrain, YStest, YDtrain, YDtest = get_data(systole, diastole, minimax, base_path, meta, case_scan_id_map)

    min = data.min(axis=0)
    max = data.max(axis=0)
    r = max - min
    r[r==0] = 0.00001

    data /= r
    data_test /= r

    print data[:5,:20]
    print YStrain[:10]

    YStrain /= 610
    YStest /= 610


    if "RF" != sys.argv[1]:
        EPOCHES = 250
        A = .008

        ss = [data.shape[1], 152, 1]
        ss2 = [data.shape[1], 132, 32, 1]
        ann = ANN(ss, 0)
        ann2 = ANN(ss2, 0)

        YStrain2 = 1. - YStrain

        for e in range(EPOCHES):
            ann.partial_fit(data, YStrain, A=A)
            ann2.partial_fit(data, YStrain2, A=A)
            print e, ann.cost.value, ann2.cost.value

    else:
        ann = RandomForestRegressor(n_estimators=100)
        ann.fit(data, YStrain)
   

    # testing
    if "RF" != sys.argv[1]:
        predictions = ann.predict_proba(data_test) 
        predictions2 = ann2.predict_proba(data_test) 
    else:
        predictions = ann.predict(data_test)
 
    E = 0.
    

    for i in range(predictions.shape[0]):
        if "RF" != sys.argv[1]:
            P = predictions[i,1]
            P2 = predictions2[i,1]
            P -= (1. - (P + P2))
        else:
            P = predictions[i]

        print P, "vs", YStest[i], "(%f vs %f)" % (P*610, YStest[i]*610)

        E += np.abs(P - YStest[i])
    E = E/YStest.shape[0]
    print "ERROR", E, "(%f)" % (E*610)
    





if __name__ == "__main__":
    main() 
