

import os
import numpy as np

from dicom_wr import DICOM

PATH_BASE = os.path.join("..", "data", "train")


cases = [d for d in os.listdir(PATH_BASE) if not d.startswith('.')]
scans = []

for d in cases:
    path = os.path.join(PATH_BASE, d, "study") 
    scans.extend([os.path.join(path, s) for s in os.listdir(path) if not s.startswith('.')])


print len(scans)


HEADER = "case,scan,MDSeriesDescription,MDPixelRepresentation,MDBitsStored,MDBitsAllocated,MDColumns,MDRows,MDGender,MDAge,MDLargestPixelValue,MDSliceThickness,MDFlipAngle,MDImagePositionPatient0,MDImagePositionPatient1,MDImagePositionPatient2,MDImageOrientationPatient0,MDImageOrientationPatient1,MDImageOrientationPatient2,MDImageOrientationPatient3,MDImageOrientationPatient4,MDImageOrientationPatient5\n"



with open(os.path.join("..", "data", "metadata.csv"), "w+") as fout:
    fout.write(HEADER)
    for dir in scans:
        tokens = dir.split(os.path.sep)
        case = tokens[-3]
        scan = tokens[-1]

        fname = os.path.join(dir, [f for f in os.listdir(dir) if not f.startswith('.')][0])

        dicom = DICOM()
        dicom.verbose(False)
        dicom.fromfile(fname)
        meta = dicom.img_metadata()

        data = "%s,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" % (
                      case,
                      scan,
                      meta[dicom.MDSeriesDescription],
                      meta[dicom.MDPixelRepresentation],
                      meta[dicom.MDBitsStored],
                      meta[dicom.MDBitsAllocated],
                      meta[dicom.MDColumns],
                      meta[dicom.MDRows],
                      meta[dicom.MDGender],
                      meta[dicom.MDAge],
                      meta[dicom.MDLargestPixelValue],
                      meta[dicom.MDSliceThickness],
                      meta[dicom.MDFlipAngle],
                      meta[dicom.MDImagePositionPatient + 0],
                      meta[dicom.MDImagePositionPatient + 1],
                      meta[dicom.MDImagePositionPatient + 2],
                      meta[dicom.MDImageOrientationPatient + 0],
                      meta[dicom.MDImageOrientationPatient + 1],
                      meta[dicom.MDImageOrientationPatient + 2],
                      meta[dicom.MDImageOrientationPatient + 3],
                      meta[dicom.MDImageOrientationPatient + 4],
                      meta[dicom.MDImageOrientationPatient + 5])
        fout.write(data)

        dicom.free()
