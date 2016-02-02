

import os
import sys
from pathes import *
from utils_wr import *
from core import *
from ann import ANN


FRAMES = 100
ROWS = 28
COLS = 28
F = 2
S = 2

EPOCHES = 600
A=.008

chanks_num = 10


def max_pool(buffer, F, S):

    rows = (ROWS - F) / S + 1
    cols = (COLS - F) / S + 1

    res = np.zeros((FRAMES, rows, cols), dtype=np.float64)

    brow = 0
    for r in range(rows):
        bcol = 0
        for c in range(cols):
            for d in range(FRAMES):
                p = buffer[d, brow : brow+F, bcol : bcol+F].max()
                res[d,r,c] = p
            bcol += S
        brow += S

    return res





def prepare_data(F, S, meta, case_scan_id_map, systole, diastole):
 
    new_rows = (ROWS - F) / S + 1
#    new_rows = ROWS
    N = 0
    for case in [d for d in os.listdir(PATH_TRAIN_F) if not d.startswith('.')]:
        path = os.path.join(PATH_TRAIN_F, case, "study")
        for scan in [d for d in os.listdir(path) if not d.startswith('.')]:
            N += 1

    indices = range(N)
    np.random.shuffle(indices)
    N = int(N * 1.)
    indices = indices[:N]
    to_train = int(N * .8)
    train_indices = set(indices[:to_train])
    indices = set(indices)

    print "num:", N, to_train

    idx       = 0
    idx_train = 0
    idx_test  = 0

    columns = FRAMES * new_rows * new_rows + 20
    #columns = 20

    t1 = to_train / chanks_num
    t4 = to_train - t1 * (chanks_num - 1)

    data = [None] * chanks_num
    for ch in range(chanks_num):
        data[ch] = np.zeros((t1, columns), dtype=np.float64)
    data[-1] = np.zeros((t4, columns), dtype=np.float64)

    counts = [t1] * chanks_num
    counts[-1] = t4
    rows_num = [0] * chanks_num
    
    data_test = np.zeros((N - to_train, columns), dtype=np.float64)

    YStrain = np.zeros((to_train,), dtype=np.float64)
    YStest = np.zeros((N-to_train,), dtype=np.float64)

    YDtrain = np.zeros((to_train,), dtype=np.float64)
    YDtest = np.zeros((N-to_train,), dtype=np.float64)


    idx = 0
    for case in [d for d in os.listdir(PATH_TRAIN_F) if not d.startswith('.')]:  #[:20]:
        ID = int(case)
        path = os.path.join(PATH_TRAIN_F, case, "study")
        for scan in [d for d in os.listdir(path) if not d.startswith('.')]:
            if idx not in indices:
                idx += 1
                continue

            path_full = os.path.join(path, scan)
            fnames = [f for f in os.listdir(path_full) if not f.startswith('.')]

            vec = np.fromfile(os.path.join(path_full, fnames[0]), dtype=np.float64, sep='')
            vec = vec.reshape((FRAMES, ROWS, COLS))
            vec = max_pool(vec, F, S)

            if idx in train_indices:
                chank = 0
                while counts[chank] == 0:
                    chank += 1
                counts[chank] -= 1

                data[chank][rows_num[chank], :-20] = vec.reshape((columns-20,))
                data[chank][rows_num[chank], -20:] = meta[case_scan_id_map[(ID, scan)]].copy()
                rows_num[chank] += 1
                 
          
                YStrain[idx_train] = systole[ID-1]
                YDtrain[idx_train] = diastole[ID-1]
                idx_train += 1
            else:
                data_test[idx_test, :-20] = vec.reshape((columns-20,))
                data_test[idx_test, -20:] = meta[case_scan_id_map[(ID, scan)]].copy()
                YStest[idx_test] = systole[ID-1]
                YDtest[idx_test] = diastole[ID-1]
                idx_test += 1
            idx += 1
    return data, data_test, YStrain, YStest, YDtrain, YDtest
    

def main():
    systole, diastole = load_train_vals()
    meta, case_scan_id_map = load_metadata()
    data, data_test, YStrain, YStest, YDtrain, YDtest = prepare_data(F, S, meta, case_scan_id_map, systole, diastole)

    MAX_VAL = 610
    YStrain /= MAX_VAL
    YStest /= MAX_VAL
    YDtrain /= MAX_VAL
    YDtest /= MAX_VAL

    print YStrain[:20]

    COLUMNS = data[0].shape[1]

    ann = None
    if 2 == len(sys.argv):
        ann, m, r = load_ann(sys.argv[1], no_mr=True)
    else:
        ss = [COLUMNS, 128, 4, 1]
        ann = ANN(ss,  0.)


    #
    #
    min1 = data[0].min(axis=0)
    min2 = data[1].min(axis=0)
    min3 = data[2].min(axis=0)
    min4 = data[3].min(axis=0)
    min = np.vstack((min1, min2, min3, min4)).min(axis=0)
    
    max1 = data[0].max(axis=0)
    max2 = data[1].max(axis=0)
    max3 = data[2].max(axis=0)
    max4 = data[3].max(axis=0)
    max = np.vstack((max1, max2, max3, max4)).max(axis=0)
    
    r = max - min
    r[r==0] = 0.00001
    data[0] -= min
    data[0] /= r
    data[1] -= min
    data[1] /= r
    data[2] -= min
    data[2] /= r
    data[3] -= min
    data[3] /= r


    data_test -= min
    data_test /= r
    #
    #


    for e in range(EPOCHES):
        chank = np.random.randint(chanks_num)
        ann.partial_fit(data[chank], YStrain, L=0, A=A)
        print e, ann.cost.value


    cnt = 0
    fname = os.path.join(PATH_CACHE, "cnn_LR.bin")
    while os.path.exists(fname):
        cnt += 1
        fname = os.path.join(PATH_CACHE, "cnn_LR_%d.bin" % cnt) 
    save_ann(ann, None, None, os.path.join(PATH_CACHE, "cnn_LR.bin"))

    # test
    cost = 0.
    for i in range(data_test.shape[0]):
        v = data_test[i].copy()
        y = YStest[i]
        if 0 == y:
            continue
        PP = ann.predict_proba(v)
        P = PP[0,1]
        print P, "vs", y, "(", P*MAX_VAL, "vs", y*MAX_VAL, ")" 
        cost += np.abs(P - y)
    E = cost / data_test.shape[0]
    print "ERROR", E, E*MAX_VAL




if __name__ == "__main__":
    main()




