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
        #cpdef na[nf64, ndim=1] update_weights_1(
        #    self, na[nf64, ndim=1] weights, na[nf64, ndim=1] gradW):
        #    return weights - self.lr*gradW
        return 
    
    cpdef na[nf64, ndim=2] update_weights(
        self, na[nf64, ndim=2] weights, na[nf64, ndim=2] gradW):
        return weights - self.lr*gradW
    
    cpdef na[nf64, ndim=1] update_biases(
        self, na[nf64, ndim=1] biases, na[nf64, ndim=1] gradB):
        return biases - self.lr*gradB
    
def sgd(lr=0.01):
    return 'SGD_{}'.format(lr)
        

cdef class MomentumSGD:
    cdef double lr, alpha
    cdef na wmoment, bmoment
    def __init__(self, double lr, double alpha):
        self.lr = lr
        self.alpha = alpha
        
    cpdef set_shape(self, na weights, na biases):
        self.wmoment = np.zeros_like(weights)
        self.bmoment = np.zeros_like(biases)
        return self.wmoment
    
    cpdef na[nf64, ndim=2] update_weights(
        self, na[nf64, ndim=2] weights, na[nf64, ndim=2] gradW):
        self.wmoment = self.alpha*self.wmoment - self.lr*gradW
        return weights + self.wmoment
    
    cpdef na[nf64, ndim=1] update_biases(
        self, na[nf64, ndim=1] biases, na[nf64, ndim=1] gradB):
        self.bmoment = self.alpha*self.bmoment - self.lr*gradB
        return biases + self.bmoment
    
    cpdef na[nf64, ndim=2] update_biases_2(
        self, na[nf64, ndim=2] biases, na[nf64, ndim=2] gradB):
        self.bmoment = self.alpha*self.bmoment - self.lr*gradB
        return biases + self.bmoment
    
def momentum_sgd(lr=0.01, alpha=0.9):
    return 'MomentumSGD_{},{}'.format(lr, alpha)