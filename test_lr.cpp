


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


Y1 = int(sys.argv[2])
Y2 = int(sys.argv[3])


AGE_IDX = 0
GENDER_IDX = 0
POSITIONS_IDX = 0



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


def img_norm(img):
    m = img.min()
    r = img.max() - m
    img -= m
    img /= r
#    img *= 255.


def filter_data(data, indices):
    return data[indices]



def get_data(systole, diastole, minimax, base_path, meta, case_scan_id_map):
    keys = minimax.keys()
    N = len(keys)
    to_train = int(N * .7)
    indices = range(N)
    np.random.shuffle(indices)
    indices_train = set(indices[:to_train])


    SHIFT = 10


    F = 4
    S = 2


    ROWS = 400
    COLS = 400

    R = 60
    C = 60


    META_SIZE = 5 + 3 + 6
    meta_indices = [4,5,6,7,10] + [11,12,13] + [14,15,16,17,18,19]

    global GENDER_IDX, AGE_IDX, POSITIONS_IDX
    GENDER_IDX = R * C + 2
    AGE_IDX = R * C + 3

    data = np.zeros((to_train, R * C + META_SIZE), dtype=np.float64)
    data_test = np.zeros((N - to_train, R * C + META_SIZE), dtype=np.float64)

    YStrain = np.zeros((to_train,), dtype=np.float64)
    YStest = np.zeros((N - to_train,), dtype=np.float64)
    YDtrain = np.zeros((to_train,), dtype=np.float64)
    YDtest = np.zeros((N - to_train,), dtype=np.float64)

    idx = 0
    idx_test = 0


    ttt = 0


    for i in range(N):
        key = keys[i]

        orig_img = spm.imread(os.path.join(base_path, str(key[0]), "study", key[1], minimax[key][0]))
        orig_img = orig_img.astype(np.float64)

        orig_img2 = spm.imread(os.path.join(base_path, str(key[0]), "study", key[1], minimax[key][1]))
        orig_img2 = orig_img2.astype(np.float64)

        img_norm(orig_img)
        img_norm(orig_img2)

        #orig_img = np.abs(orig_img2 - orig_img)

#        if ttt == 0:
#            ttt = 1
#            plt.clf()
#            plt.imshow(orig_img)
#            plt.show()


        img = orig_img
        img = utils.img_resize(img, R, C)


        if i in indices_train:
            if META_SIZE:
                data[idx,:-META_SIZE] = img.flatten()
                data[idx,-META_SIZE:] = meta[case_scan_id_map[key]][meta_indices]
                #data[idx,-META_SIZE:] = orig_img.shape + orig_img.shape
            else:
#                if ttt < 10:
#                    ttt += 1
#                    plt.clf()
#                    plt.imshow(img)
#                    plt.show()
                data[idx] = img.flatten()
 
            YStrain[idx] = systole[key[0]-1]
            YDtrain[idx] = diastole[key[0]-1]
            idx += 1

        else:
            if META_SIZE:
                data_test[idx_test,-META_SIZE:] = meta[case_scan_id_map[key]][meta_indices]
                #data_test[idx_test,-META_SIZE:] = orig_img.shape + orig_img.shape
                data_test[idx_test,:-META_SIZE] = img.flatten()
            else:
                data_test[idx_test] = img.flatten()

            YStest[idx_test] = systole[key[0]-1]
            YDtest[idx_test] = diastole[key[0]-1]
            idx_test += 1

    #if META_SIZE:
        #min = data[:,-META_SIZE:].min(axis=0)
        #max = data[:,-META_SIZE:].max(axis=0)
        #r = max - min

        #data[:,-META_SIZE:][r == 0] = 0
        #data_test[:,-META_SIZE:][r == 0] = 0

        #r[r==0] = 0.00001

        #r = 1000
  
        #data[:,-META_SIZE:] /= r
        #data_test[:,-META_SIZE:] /= r

    
    return data, data_test, YStrain, YStest, YDtrain, YDtest






def main():
    np.random.seed()

    base_path = PATH_TRAIN_H

    N = get_N(base_path)
    print N

    systole, diastole = load_train_vals()

    minimax = get_minimax()
    #print minimax

    meta, case_scan_id_map = load_metadata()

    data, data_test, YStrain, YStest, YDtrain, YDtest = get_data(systole, diastole, minimax, base_path, meta, case_scan_id_map)

    ii = (12*Y1 < data[:,AGE_IDX]) & (data[:,AGE_IDX] <= 12*Y2) 
    print ii.shape
    data = filter_data(data, ii)
    print data.shape
    YStrain = filter_data(YStrain, ii)
    YDtrain = filter_data(YDtrain, ii)

    data[:,GENDER_IDX][data[:,GENDER_IDX] != 1.] = -1.
    min = data[:,AGE_IDX+1:].min(axis=0)
    max = data[:,AGE_IDX+1:].max(axis=0)
    r = max - min
    data[:,AGE_IDX+1:][:,r==0] = 0
    r[r==0] = 0.00001
    data[:,AGE_IDX+1:] /= r



    ii = ((12*Y1) < data_test[:,AGE_IDX]) & (data_test[:,AGE_IDX] <= (12*Y2)) 
    data_test = data_test[ii]
    YStest = YStest[ii]
    YDtest = YDtest[ii]
    data_test[:,GENDER_IDX][data_test[:,GENDER_IDX] != 1.] = -1.
    data_test[:,AGE_IDX+1:][:,r==0.00001] = 0
    data_test[:,AGE_IDX+1:] /= r

    print data[:5,:]
    print YStrain[:]

    YStrain /= 610
    YStest /= 610

    print data.shape
    print data_test.shape

    if "RF" != sys.argv[1]:
        EPOCHES = 3000
        A = .02

        ss = [data.shape[1], 32, 1]
        ann = ANN(ss, 0)

#        ww, bb = ann.get_weights()
#        np.random.seed()
#        ww = np.random.rand(ww.shape[0])
#        bb = np.random.rand(bb.shape[0])
#        ann.set_weights(ww, bb)

        

        for e in range(EPOCHES):
            ann.partial_fit(data, YStrain, n_iter=1, A=A)
            print e, ann.cost.value    #, ann2.cost.value

    else:
        ann = RandomForestRegressor(n_estimators=500)
        ann.fit(data, YStrain)
   

    # testing
    if "RF" != sys.argv[1]:
        predictions = ann.predict_proba(data_test) 
    else:
        predictions = ann.predict(data_test)
 
    E = 0.
    

    for i in range(predictions.shape[0]):
        if "RF" != sys.argv[1]:
            P = predictions[i,1]
        else:
            P = predictions[i]

        print P, "vs", YStest[i], "(%f vs %f)" % (P*610, YStest[i]*610)

        E += np.abs(P - YStest[i])
    E = E/YStest.shape[0]
    print "ERROR", E, "(%f)" % (E*610)

    print data.shape, data_test.shape

    





if __name__ == "__main__":
    main() 

