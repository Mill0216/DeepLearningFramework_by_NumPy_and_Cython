import cython
cimport cython
import numpy as np
cimport numpy as np

ctypedef np.float_t nf64
ctypedef np.ndarray na


cdef class MSE:
    cdef:
        na dif_sum
        int batchsize_count
        list input_shape
    def __init__(self):
        self.batchsize_count = 0
        
    cpdef set_params(self, list input_shape):
        self.input_shape = input_shape
        self.dif_sum = np.zeros(tuple(input_shape))
    
    cpdef nf64 forward(self, na pred, na y):
        cdef na dif
        dif = pred-y
        self.dif_sum += dif
        self.batchsize_count += 1
        return np.mean(np.square(dif))
        
    cpdef na backward(self):
        cdef na delta
        delta = self.dif_sum/self.batchsize_count
        self.dif_sum = np.zeros(tuple(self.input_shape))
        self.batchsize_count = 0
        return delta


cdef class MAE:
    cdef:
        na dif_na
        list input_shape
    
    cpdef set_params(self, list input_shape):
        self.input_shape = input_shape
        
    cpdef nf64 forward(self, na pred, na y):
        self.dif_na = pred-y
        return np.mean(np.abs(self.dif_na))
        
    cpdef na backward(self):
        return (self.dif_na>0).astype(np.int)*2-1
