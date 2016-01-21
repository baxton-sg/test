

import os
import sys
import csv
import numpy as np
import scipy as sp
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from dicom_wr import DICOM
from utils_wr import UTILS



utils = UTILS()


SEP = os.path.sep

dir = ""
fnames = None
files_num = 0

buffers = {}

POSITIONS = {}


NOSHOW = False
EPOCHES = 2
MEAN_MUL = 4.
FRAME_SIZE = 2
KOEF = .5
LOW_VAL = 10
HIGH_VAL = 16


def load_positions():
    global POSITIONS

    with open('positions.csv', "r") as fin:
        g = csv.reader(fin, delimiter=',')
        for tokens in g:
            #    0    1      2     3      4    5       6
            # case, sax, fname, left, right, top, bottom
            key = "%s%sstudy%s%s" % (tokens[0], SEP, SEP, tokens[1])
            pos = ( int(tokens[5]), int(tokens[6]), int(tokens[3]), int(tokens[4]), 0 )
            POSITIONS[key] = pos




def area_of_intersection(t1, b1, l1, r1,  t2, b2, l2, r2):
    return max(0, min(r1, r2) - max(l1, l2)) * max(0, min(b1, b2) - max(t1, t2))

def area(t, b, l, r):
    return (b - t) * (r - l)




def F1(p, r):
    return 2. * (p * r) / float(p + r)



def detect_rectangle(buffer, start_r, start_c, visited):
    R = buffer.shape[0]
    C = buffer.shape[1]

    top = start_r
    bottom = top

    left = start_c
    right = left
    
    S = 1

    key = (start_r, start_c)
    Q = [key]

    while 0 < len(Q):
        key = Q.pop()
        visited.add(key)

        # top
        if 0 < key[0]:
           r = key[0] - 1
           c = key[1]
           if 0 < buffer[r,c]:
               k = (r,c)
               if not k in visited:
                   S += 1
                   visited.add(k)
                   Q.append(k)

                   if top > r:
                       top = r
                   if bottom < r:
                       bottom = r
 
        # bottom
        if R > key[0] + 1:
           r = key[0] + 1
           c = key[1]
           if 0 < buffer[r,c]:
               k = (r,c)
               if not k in visited:
                   S += 1
                   visited.add(k)
                   Q.append(k)

                   if top > r:
                       top = r
                   if bottom < r:
                       bottom = r

        # left
        if 0 < key[1]:
           r = key[0]
           c = key[1] - 1
           if 0 < buffer[r,c]:
               k = (r,c)
               if not k in visited:
                   S += 1
                   visited.add(k)
                   Q.append(k)

                   if left > c:
                       left = c
                   if right < c:
                       right = c
 
        # right
        if C > key[1] + 1:
           r = key[0]
           c = key[1] + 1
           if 0 < buffer[r,c]:
               k = (r,c)
               if not k in visited:
                   S += 1
                   visited.add(k)
                   Q.append(k)

                   if left > c:
                       left = c
                   if right < c:
                       right = c
 
    return top, bottom, left, right, S







def get_rectangles(buffer):
    rects = []
    visited = set()

    R = buffer.shape[0]
    C = buffer.shape[1]

    for r in range(R):
        for c in range(C):
            if 0 == buffer[r,c]:
                continue

            key = (r,c)
            if key in visited:
                continue

            top, bottom, left, right, S = detect_rectangle(buffer, r, c, visited)
            rects.append((top, bottom, left, right, S))

    return rects




def process_and_save(): 
    global dir, fnames, files_num, buffers

    path_base = "..%sdata%s" % (SEP, SEP)

    dicom.VERBOSE = False

    for d in [d for d in os.listdir(path_base) if not d.startswith(".")]:
        path1 = path_base + d + "%sstudy%s" % (SEP, SEP)
        for d2 in [d for d in os.listdir(path1) if not d.startswith(".")]:
            path_final = path1 + d2 + SEP
            dir = path_final
            

            fnames = [f for f in os.listdir(dir) if "dcm" in f and not f.startswith('.')]
            files_num = len(fnames)

            buffers.clear()


            fig = process(ret=True)

            tokens = dir.split(SEP)

            fname = "..%simgs%s%s_%s_%s.png" % (SEP, SEP, tokens[-4], tokens[-2], tokens[-1])
            fig.savefig(fname)
   






def process(ret=False): 
    global dir, fnames, files_num, buffers


    fnames = [f for f in os.listdir(dir) if "dcm" in f and not f.startswith('.')]
    files_num = len(fnames)

    keys = []
    for i, f in enumerate(fnames):
        key = dir + f
        keys.append(key)
        if key not in buffers:
            dicom = DICOM()
            dicom.verbose(False)
            dicom.fromfile(key)
            b = dicom.img_buffer()
            buffers[key] = b
            dicom.free()

   
     


    prev = buffers[keys[0]]
    shape2D = prev.shape
    shape3D = (shape2D[0], shape2D[1], 1)

#    freq = np.zeros(shape2D, dtype=float)
#    #freq = np.ones(shape2D, dtype=float)
#
#    cnt = 0
#    thr = 1
#
#    for i in range(1, files_num):
#        cnt += 1
#        if cnt != thr:
#            continue
#        cnt = 0
#        curr = buffers[keys[i]]
#        #
#        tmp = (prev - curr) ** 2
#        mv = np.mean(tmp)
#        std = np.std(tmp)
#        zz = tmp - mv
#        tmp[zz < mv + std*MEAN_MUL] = 0
#        freq[tmp != 0] += 1
#        #
#        prev = curr
#
#     
#
#    freq[freq < 3] = 0
#    freq[freq > 6] = 0

    frames = np.zeros((files_num, shape2D[0], shape2D[1]), dtype=np.float64)
    for f in range(files_num):
        frames[f,:,:] = buffers[keys[f]]
    freq = utils.get_frequencies(frames, MEAN_MUL, LOW_VAL, HIGH_VAL)
    
    freq = utils.filter(freq, FRAME_SIZE, KOEF)        

    shape2D = (freq.shape[0], freq.shape[1])
    shape3D = (freq.shape[0], freq.shape[1], 1)
     

#    for e in range(EPOCHES):
#        R = shape2D[0]
#        C = shape2D[1]
#        S = FRAME_SIZE
#        K = KOEF
#        res = np.zeros(shape2D, dtype=float)
#        for r in range(R):
#            if (r + S) >= R:
#                continue
#            for c in range(C):
#                if (c + S) >= C:
#                    continue
#                tmp = freq[r : r + S, c : c + S]
#                #mv = np.mean(tmp)
#                mv = tmp.sum() / (S**2)
#    
#                if K > mv:
#                    freq[r+S/2,c+S/2] = 0
#                else:
#                    freq[r+S/2,c+S/2] = tmp.mean()
#   
#    freq[:, :10] = 0
#    freq[:, -10:] = 0
#    freq[:10, :] = 0
#    freq[-10:, :] = 0


    rects = get_rectangles(freq)


    freq = freq.reshape(shape3D)

    
    tokens = dir.split(SEP)
    #print tokens
    key = "%s%sstudy%s%s" % (tokens[-4], SEP, SEP, tokens[-2])
    pos = POSITIONS[key] if key in POSITIONS else None
    if None != pos:
        SP = area(pos[0], pos[1], pos[2], pos[3])
        #print "real", pos



    if not NOSHOW:
        plt.clf()
        ax = plt.subplot(121)
        plt.imshow(buffers[keys[0]]*100)
 
        max_r = None
        max_P = 0
       
        for r in rects:
            SR = area(r[0], r[1], r[2], r[3])
            SI = 0
            if None != pos:
                SI = area_of_intersection(pos[0], pos[1], pos[2], pos[3],  r[0], r[1], r[2], r[3])

            if 0 == SI:
                P = 0
            else:
                P = F1(float(SI) / SP, float(SI) / SR)

            if P > max_P:
                max_P = P
                max_r = r

            ax.add_patch(patches.Rectangle((r[2], r[0]), r[3] - r[2], r[1] - r[0], fill=False, linewidth=2, edgecolor='red'))

            print "rect", P, str(r)

        if None != pos:
            ax.add_patch(patches.Rectangle((pos[2], pos[0]), pos[3] - pos[2], pos[1] - pos[0], fill=False, linewidth=2, edgecolor='green'))


        print "MAX:", max_P, max_r
        ax.add_patch(patches.Rectangle((max_r[2], max_r[0]), max_r[3] - max_r[2], max_r[1] - max_r[0], fill=False, linewidth=2, edgecolor='yellow'))


        plt.subplot(122)
        plt.imshow(np.concatenate((freq,freq,freq), axis=2))

        if not ret:
            plt.show()
        else:
            return plt.gcf()

    else:
        max_P = 0
 
        for r in rects:
            SR = area(r[0], r[1], r[2], r[3])
            SI = area_of_intersection(pos[0], pos[1], pos[2], pos[3],  r[0], r[1], r[2], r[3])

            if 0 == SI:
                P = 0
            else:
                P = F1(float(SI) / SP, float(SI) / SR)

            if max_P < P:
                max_P = P

        return max_P



def run_many():
    global dir, NOSHOW, EPOCHES, MEAN_MUL, FRAME_SIZE, KOEF

    NOSHOW = True

    path_base = "..%sdata%strain%s" % (SEP, SEP, SEP)

    md = .5

    max_res = 0
    max_data = None

    

    for EPOCHES in range(0, 53):
        MEAN_MUL = 1.
        for t1 in range(5):
            MEAN_MUL *= 2
            for FRAME_SIZE in range(2, 16):
                KOEF = 1.
                for t2 in range(5):
                    KOEF *= 2

                    res = 0.
                    cnt = 0
                    for d in [d for d in os.listdir(path_base) if not d.startswith(".")][0]:
                        path1 = path_base + d + "%sstudy%s" % (SEP, SEP)
                        for d2 in [d for d in os.listdir(path1) if not d.startswith(".")]:
                            path_final = path1 + d2 + SEP
                            dir = path_final

                            res += process()
                            cnt += 1

                    data = (EPOCHES, MEAN_MUL, FRAME_SIZE, KOEF)
                    res /= cnt
                    print res, str(data)

                    if max_res < res:
                        max_res = res
                        max_data = data
    print "MAX:", max_res, max_data



def main(): 
    global dir, fnames, files_num, buffers

    dir = sys.argv[1]

    if 2 < len(sys.argv):
        global NOSHOW, EPOCHES, MEAN_MUL, FRAME_SIZE, KOEF
        NOSHOW = True if sys.argv[2] == "True" else False
        EPOCHES = int(sys.argv[3])
        MEAN_MUL = float(sys.argv[4])
        FRAME_SIZE = int(sys.argv[5])
        KOEF = float(sys.argv[6])

    if dir == "ALL":
        process_and_save()
        return 0

    if dir == "STAT":
        get_stat()
        return 0

    load_positions()


    fnames = [f for f in os.listdir(dir) if "dcm" in f and not f.startswith('.')]
    files_num = len(fnames)



    if not NOSHOW:
        process() 
    else:
        print process()






if __name__ == "__main__":
    main()


