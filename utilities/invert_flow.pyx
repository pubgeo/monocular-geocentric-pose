#
# Fast cython function to invert a flow field
#

#cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
import cv2

def invert_flow(int[:,:,:] flow, float[:,:] mag, float fac):
    cdef int cols = flow.shape[0]
    cdef int rows = flow.shape[1]
    # do not stretch magnitude more than 1/fac to avoid empty output pixels
	# good values are 0.5 for up to 2x or 0.25 for up to 4x
    cdef float factor = fac
    cdef int colsf = int(cols*factor)
    cdef int rowsf = int(rows*factor)
    cdef float[:,:] new_mag = np.zeros((colsf,rowsf), dtype=np.float32)
    cdef int[:,:] new_mask = np.ones((colsf,rowsf), dtype=np.int32) * 255
    cdef int[:,:,:] new_flow = np.zeros((colsf,rowsf,2), dtype=np.int32)
    cdef float_flow = np.zeros((colsf,rowsf,2), dtype=np.float32)
    cdef out_flow = np.zeros(flow.shape, dtype=np.float32)
    cdef int_mask = np.zeros((colsf,rowsf), dtype=np.uint8)
    cdef out_mask = np.zeros((cols,rows), dtype=np.uint8)
    cdef int i, j, i2, j2
    cdef int keep
    for i in range(0,cols):
        for j in range(0,rows):
            i2 = int((i+flow[i,j,1])*factor)
            if (i2 < 0) | (i2 > colsf-1):
                continue
            j2 = int((j+flow[i,j,0])*factor)
            if (j2 < 0) | (j2 > rowsf-1):
                continue
            # when multiple input pixels map to the same output pixel, keep the one on top
            keep = mag[i,j] >= new_mag[i2,j2]
            if (keep == 0):
                continue
            new_mag[i2,j2] = mag[i,j]
            new_flow[i2,j2,:] = flow[i,j,:]
            new_mask[i2,j2] = 0
    float_flow = np.copy(new_flow).astype(np.float32)
    out_flow = cv2.resize(float_flow,(cols,rows))
    int_mask = np.copy(new_mask).astype(np.uint8)
    out_mask = cv2.resize(int_mask,(cols,rows))
    return out_flow, out_mask

