

import os
import sys
import ctypes
import numpy as np

UTILS_DLL = ctypes.cdll.LoadLibrary("utils2.dll")




class UTILS(object):
    def __init__(self):
        pass


    def get_statistics(self, buffer):
        tmp = buffer
        if tmp.dtype != np.float64:
            tmp = tmp.astype(np.float64)

        total_size = tmp.shape[0]
        for s in range(1, len(tmp.shape)):
            total_size *= tmp.shape[s]

        mv = ctypes.c_double(0)
        skew = ctypes.c_double(0)
        var = ctypes.c_double(0)
        kur = ctypes.c_double(0)

        UTILS_DLL.get_statistics(ctypes.c_void_p(tmp.ctypes.data),
                                 ctypes.c_int(total_size),
                                 ctypes.c_void_p(ctypes.addressof(mv)),
                                 ctypes.c_void_p(ctypes.addressof(skew)),
                                 ctypes.c_void_p(ctypes.addressof(var)),
                                 ctypes.c_void_p(ctypes.addressof(kur)))
        return mv, skew, var, kur



    def get_frequencies(self, frames, MEAN_MUL, LOW_VAL, HIGH_VAL):
        rows = frames.shape[1]
        cols = frames.shape[2]
        frames_num = frames.shape[0]

        frames = frames.flatten()
        freq = np.zeros((rows * cols, ), dtype=np.float64)

        UTILS_DLL.get_frequencies(ctypes.c_void_p(frames.ctypes.data),
                                  ctypes.c_int(rows),
                                  ctypes.c_int(cols),
                                  ctypes.c_int(frames_num),
                                  ctypes.c_void_p(freq.ctypes.data),
                                  ctypes.c_double(MEAN_MUL),
                                  ctypes.c_double(LOW_VAL),
                                  ctypes.c_double(HIGH_VAL))
        return freq.reshape((rows, cols))


    def filter(self, freq, F, S, K):
        tmp = freq
        if freq.dtype != np.float64:
            tmp = tmp.astype(np.float64)

        rows = freq.shape[0]
        cols = freq.shape[1]

        new_rows = (rows - F) / S + 1
        new_cols = (cols - F) / S + 1

        if (float(rows - F) / S + 1) != new_rows:
            raise Exception("Wrong Frame (%d) and Stride (%d) combination, (%d,%d)" % (F, S, rows, cols))

        new_freq = np.zeros((new_rows, new_cols), dtype=np.float64)

        UTILS_DLL.filter(ctypes.c_void_p(tmp.ctypes.data),
                         ctypes.c_int(rows),
                         ctypes.c_int(cols),
                         ctypes.c_int(F),
                         ctypes.c_int(S),
                         ctypes.c_double(K),
                         ctypes.c_void_p(new_freq.ctypes.data))
        return new_freq









def main():
    from dicom_wr import DICOM
    import scipy.stats as stats

    fname = os.path.join("..", "data", "train", "1", "study", "sax_10", "IM-4562-0001.dcm")
    print "processing", fname

    dicom = DICOM()
    dicom.verbose(False)
    dicom.fromfile(fname)
    buffer = dicom.img_buffer()
    dicom.free()
    dicom = None

    utils = UTILS()
    mv, skew, var, kur = utils.get_statistics(buffer)

    print mv.value, skew.value, var.value, kur.value
    print np.mean(buffer), stats.skew(buffer.flatten()), np.var(buffer), stats.kurtosis(buffer.flatten())




if __name__ == "__main__":
    main()
     


