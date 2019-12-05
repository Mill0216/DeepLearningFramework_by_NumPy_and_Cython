import cython
cimport cython
import numpy as np
cimport numpy as np
import copy

from Layers import *
from Optimizers import *

ctypedef np.float_t nf64
ctypedef np.ndarray na

#managerクラスに層(重み含)やオプティマイザの情報を入れる。
#層以外は好きな時に変えれるようにしたいね。

cdef class manager():
    cdef int i_ndim, o_ndim
    cdef list layers
    cdef optimizer, loss_func
    
    def __init__(self, list input_size, 
                 list layers, optimizer, update_freq, loss_func):
        self.layers = []
        for layer_info in layers:
            options = layer_info.split('_')
            exec('layer = {}({})'.format(options[0], options[1]), globals())
            self.layers.append(layer)
        
        options = optimizer.split('_')
        exec('_optimizer = {}({})'.format(options[0], options[1]), globals())
        exec('_loss_func = {}()'.format(loss_func), globals())
        self.optimizer = _optimizer
        self.loss_func = _loss_func
        self.layer_setup(input_size)
    cdef layer_setup(self, list input_size):
        cdef list pre_output = input_size
        for layer in self.layers:
            if layer.not_activation:
                pre_output = layer.set_params(pre_output)
                layer.set_optimizer(copy.copy(self.optimizer))
        self.loss_func.set_params(pre_output)
            
    cpdef double train_online(self, list x_data, list y_data):
        cdef list losses = []
        cdef int data_size = len(x_data)
        #cython、zip使えない説(forのコンマでコンパイルエラー)
        for i in range(data_size):
            x = x_data[i]
            y = y_data[i]
            losses.append(self.loss_calc(self.forward(x), y))
            self.backward()
        return sum(losses)/data_size
            
    cpdef double train_batch(self, list x_data, list y_data):
        cdef nf64 return_loss = self.test_batch(x_data, y_data)
        self.backward()
        return return_loss
        
    cpdef double test_batch(self, list x_data, list y_data):
        cdef list losses = []
        cdef int data_size = len(x_data)
        for i in range(data_size):
            x = x_data[i]   
            y = y_data[i]
            losses.append(self.loss_calc(self.forward(x), y))
        return sum(losses)/data_size
    
    cpdef na predict_batch(self, list x_data):
        cdef list pred_list = []
        cdef na pred
        for x in x_data:
            pred = self.forward(x)
            pred_list.append(pred)
        return np.array(pred_list)
        
    cdef na forward(self, na x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
        
    cdef nf64 loss_calc(self, na pred, na y):
        return self.loss_func.forward(pred, y)
        
    cdef backward(self):
        cdef na delta = self.loss_func.backward()
        for layer in self.layers[::-1]:
             delta = layer.backward(delta)
                
    pass