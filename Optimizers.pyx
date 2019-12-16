import cython
cimport cython
import numpy as np
cimport numpy as np

ctypedef np.float_t nf64
ctypedef np.ndarray na


cdef class SGD:
    cdef double lr
    cdef int wndim, bndim
    def __init__(self, double lr):
        self.lr = lr
        
    cpdef set_shape(self, weights, biases):
        return 
    
    cpdef na update_weights(
            self, na weights, na gradW):
        return weights - self.lr*gradW
    
    cpdef na[nf64, ndim=1] update_biases(
        self, na[nf64, ndim=1] biases, na[nf64, ndim=1] gradB):
        return biases - self.lr*gradB
        

cdef class MomentumSGD:
    cdef double lr, alpha
    cdef public na weights, biases
    def __init__(self, double lr, double alpha):
        self.lr = lr
        self.alpha = alpha
        
    cpdef set_shape(self, na weights, na biases):
        self.weights = np.zeros((*np.array(weights).shape))
        self.biases = np.zeros((*np.array(biases).shape))
        return self.weights
    
    cpdef na update_weights(self, na weights, na gradW):
        self.weights = self.alpha*self.weights - (1-self.alpha)*self.lr*gradW
        return weights + self.weights
    
    cpdef na[nf64, ndim=1] update_biases(
        self, na[nf64, ndim=1] biases, na[nf64, ndim=1] gradB):
        self.biases = self.alpha*self.biases - (1-self.alpha)*self.lr*gradB
        return biases + self.biases
