import cython
cimport cython
import numpy as np
cimport numpy as np
import copy
import json

ctypedef np.float_t nf64
ctypedef np.ndarray na



cdef class manager():
    cdef int i_ndim, o_ndim
    cdef list layers
    cdef optimizer, loss_func
    
    def __init__(self, list input_size, 
                 list _layers, _optimizer, _loss_func):
        self.layers = _layers
        self.optimizer = _optimizer
        self.loss_func = _loss_func
        self.layer_setup(input_size)
    cdef layer_setup(self, list input_size):
        cdef list pre_output = input_size
        cdef int i = 0
        for layer in self.layers:
            pre_output = layer.set_params(pre_output, i)
            i += 1
            if layer.not_activation:
                layer.set_optimizer(copy.copy(self.optimizer))
        self.loss_func.set_params(pre_output)
            
    cpdef double train_online(self, list x_data, list y_data, parent_directory):
        cdef list losses = []
        cdef int data_size = len(x_data)
        for i in range(data_size):
            x = x_data[i]
            y = y_data[i]
            losses.append(self.loss_calc(self.forward(x), y))
            self.backward()
        self.save_parameters(10, parent_directory)
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

    def save_parameters(self, int epoch, parent_directory):
        save_dictionary = {}
        for layer in self.layers:
            layer_info = {}
            if layer.not_activation:
                layer_info['weights'] = list(layer.weights.flatten())
                layer_info['biases'] = list(layer.biases)
                if self.optimizer.__class__.__name__ != 'SGD':
                    layer_info['opt_weights'] = list(layer.optimizer.weights.flatten())
                    layer_info['opt_biases'] = list(layer.optimizer.biases)
            save_dictionary[str(layer.layer_id)] = layer_info
        with open('{}/epoch{}.json'.format(parent_directory, epoch), 'w') as f:
            json.dump(save_dictionary, f)
            
    def load_parameters(self, int epoch, parent_directory):
        with open('{}/epoch{}.json'.format(parent_directory, epoch), 'r') as f:
            save_dictionary = json.load(f)
        for i, layer in enumerate(self.layers):
            if layer.not_activation:
                layer_info = save_dictionary[str(i)]
                layer.weights = np.array(layer_info['weights']).reshape(layer.weights.shape)
                layer.biases = np.array(layer_info['biases']).reshape(layer.biases.shape)
                layer.optimizer.weights = np.array(layer_info['opt_weights']).reshape(layer.weights.shape)
                layer.optimizer.biases = np.array(layer_info['opt_biases']).reshape(layer.biases.shape)
                
    pass
