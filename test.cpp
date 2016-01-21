

import os
import sys
import csv
import numpy as np


from dicom_wr import DICOM
from utils_wr import UTILS



utils = UTILS()


SEP = os.path.sep

dir = ""
fnames = None
files_num = 0

buffers = {}

POSITIONS = {}


# params to optimize
NOSHOW = False
EPOCHES = 1
MEAN_MUL = 2
FRAME_SIZE = 8
STRIDE = 1
KOEF = 4
LOW_VAL = 8
HIGH_VAL = 32
CUTOFF = 2.6






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





def process(): 
    global dir, fnames, files_num, buffers


    fnames = [f for f in os.listdir(dir) if "dcm" in f and not f.startswith('.')]
    files_num = len(fnames)

    origR = 0
    origC = 0

    keys = []
    for i, f in enumerate(fnames):
        key = dir + f
        keys.append(key)
        if key not in buffers:
            dicom = DICOM()
            dicom.verbose(False)
            dicom.fromfile(key)
            b = dicom.img_buffer()
            b[:30,:]=0
            b[-30:,:]=0
            b[:,:30]=0
            b[:,-30:]=0
            mv = b.mean()
            origR = b.shape[0]
            origC = b.shape[1]
            buffers[key] = utils.filter(b, 8, 2, mv/CUTOFF) 

            dicom.free()

        else:
            origR = buffers[keys[0]].shape[0]
            origC = buffers[keys[0]].shape[1]

   
     


    prev = buffers[keys[0]]
    shape2D = prev.shape
    shape3D = (shape2D[0], shape2D[1], 1)

    newR = shape2D[0]
    newC = shape2D[1]

    step = 1
    frames = np.zeros((files_num / step, shape2D[0], shape2D[1]), dtype=np.float64)
    for f in range(0, files_num, step):
        frames[f/step,:,:] = buffers[keys[f]]
    freq = utils.get_frequencies(frames, MEAN_MUL, LOW_VAL, HIGH_VAL)
    

    for e in range(EPOCHES):    
        freq = utils.filter(freq, FRAME_SIZE, STRIDE, KOEF)        

    KH = float(freq.shape[0]) / shape2D[0]
    KW = float(freq.shape[1]) / shape2D[1]

    shape2D = (freq.shape[0], freq.shape[1])
    shape3D = (freq.shape[0], freq.shape[1], 1)
     

    rects = get_rectangles(freq)


    
    tokens = dir.split(SEP)
    key = "%s%sstudy%s%s" % (tokens[-4], SEP, SEP, tokens[-2])
    pos = POSITIONS[key] if key in POSITIONS else None
    if None != pos:
        PKH = float(newR) / origR
        PKW = float(newC) / origC
        pos = (pos[0]*PKH, pos[1]*PKH, pos[2]*PKW, pos[3]*PKW)
        SP = area(pos[0], pos[1], pos[2], pos[3])
#        print pos



    max_r = None
    max_P = 0
       
    for r_row in rects:
        r = (r_row[0] / KH, r_row[1] / KH, r_row[2] / KW, r_row[3] / KW)
        SR = area(r[0], r[1], r[2], r[3])
        SI = 0
        if None != pos:
            SI = area_of_intersection(pos[0], pos[1], pos[2], pos[3],  r[0], r[1], r[2], r[3])

        if 0 == SI:
            P = 0
        else:
            P = F1(float(SI) / SP, float(SI) / SR)

#        print P, r

        if P > max_P:
            max_P = P
            max_r = r


    return max_P






def cost():
    global dir   #, NOSHOW, EPOCHES, MEAN_MUL, FRAME_SIZE, KOEF, LOW_VAL, HIGH_VAL

    path_base = "..%sdata%strain%s" % (SEP, SEP, SEP)

    max_res = 0
    max_data = None
    

    res = 0.
    cnt = 0
    for d in [d for d in os.listdir(path_base) if not d.startswith(".")][0]:
        path1 = path_base + d + "%sstudy%s" % (SEP, SEP)
        for d2 in [d for d in os.listdir(path1) if not d.startswith(".")]:
            path_final = path1 + d2 + SEP
            dir = path_final

            res += process()
            cnt += 1

    #data = (EPOCHES, MEAN_MUL, FRAME_SIZE, KOEF, LOW_VAL, HIGH_VAL)
    res /= cnt
    #print res, str(data), "(processed %d scans)" % cnt

    return res






def main(): 
    global CUTOFF

    load_positions()

    data = (EPOCHES, MEAN_MUL, FRAME_SIZE, KOEF, LOW_VAL, HIGH_VAL, CUTOFF)
    E = cost()
    print E, data

    e = 0.00001
    a = 0.01
   
    for i in range(1):
        prev_cutoff = CUTOFF
#        CUTOFF += e
        newE = cost()
        print newE, CUTOFF
        CUTOFF = prev_cutoff
        g = (newE - E) / e

        CUTOFF = CUTOFF - a * g
 
        E = cost()
        data = (EPOCHES, MEAN_MUL, FRAME_SIZE, KOEF, LOW_VAL, HIGH_VAL, CUTOFF)
        print E, data
                





if __name__ == "__main__":
    main()



