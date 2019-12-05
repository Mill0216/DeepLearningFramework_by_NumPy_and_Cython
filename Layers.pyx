import cython
cimport cython
import numpy as np
cimport numpy as np

ctypedef np.float_t nf64
ctypedef np.ndarray na

cdef class Dense:
    cdef na weights, biases, x, gradW, gradB
    cdef char* init_func
    cdef int noutput
    cdef bint use_bias
    cdef public bint not_activation
    cdef optimizer
    
    def __init__(self, int noutput, 
                 str init_func, bint use_bias=False):
        self.noutput = noutput
        init_func_e = init_func.encode('UTF-8')
        self.init_func = init_func_e
        self.use_bias = use_bias
        self.not_activation = True
        
    cpdef list set_params(self, list ninput):
        #Cythonの文字列周りは勉強せな
        #if self.init_func.decode('UTF-8', 'strict') == 'Gaussian':
        #    self.weights = np.random.randn(ninput, self.noutput)
        #    self.biases = np.random.randn(self.noutput)\
        #        if self.use_bias else np.zeros(self.noutput)
        #else:
        #    raise NotImplementedError('this library did not know '\
        #                    'the function called {}. '\
        #                'Chose Gaussian.'.format(
        #             self.init_func.decode('UTF-8', 'strict')))
        self.weights = np.random.randn(ninput[0], self.noutput)
        self.biases = np.random.randn(self.noutput)\
                if self.use_bias else np.zeros(self.noutput)
        return [self.noutput]
            
    cpdef set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer.set_shape(self.weights, self.biases)
        
    cpdef na[nf64, ndim=1] forward(self, na[nf64, ndim=1] x):
        self.x = x
        cdef na[nf64, ndim=2] WT = self.weights.T
        return np.dot(WT, x) + self.biases
    
    cpdef na[nf64, ndim=1] backward(self,na delta):
        self.gradW = mk_grad(delta, self.x)
        self.gradB = delta
        cdef na dx = np.dot(delta, self.weights.T)
        self.update_parameters()
        return dx

    cpdef update_parameters(self):
        self.weights = self.optimizer.update_weights(self.weights, self.gradW)
        if self.use_bias:
            self.biases = self.optimizer.update_biases(self.biases, self.gradB)
        
def dense(output, init_func='Gaussian', use_bias=True):
    return "Dense_{},'{}',{}".format(output, init_func, use_bias)

cdef na[nf64, ndim=2] mk_grad(na delta, na x):
    mat_delta = np.matrix(delta)
    mat_x = np.matrix(x)
    return np.array(np.dot(mat_x.T, mat_delta))


cdef class Conv2D:
    cdef int kw, kh, st_x, st_y, im_k, im_w, im_h, new_w, new_h, col1_w, col1_h, out_c
    cdef na weights, biases, x, gradW, gradB
    cdef bint use_bias
    cdef public bint not_activation
    cdef optimizer
    def __init__(self, int kw, int kh, int st_x, int st_y, int out_c, bint use_bias):
        self.kw = kw     #カーネル
        self.kh = kh
        self.st_x = st_x #stride
        self.st_y = st_y
        self.out_c = out_c
        self.use_bias = use_bias
        self.not_activation = True
        
    cpdef list set_params(self, list input_shape):
        self.im_k, self.im_w, self.im_h = input_shape
        self.new_w = int((self.im_w - self.kw) / self.st_x) + 1  
        self.new_h = int((self.im_h - self.kh) / self.st_y) + 1
        self.col1_w, self.col1_h = self.kw*self.kh, self.new_w*self.new_h
        
        self.weights = np.random.randn(self.im_k,self.col1_w,self.out_c)
        self.biases = np.zeros(self.out_c)
        return [self.out_c, self.new_w, self.new_h]
    
    cpdef set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer.set_shape(self.weights, self.biases)
        
    cpdef update_parameters(self):
        self.weights = self.optimizer.update_weights(self.weights, self.gradW)
        if self.use_bias:
            self.biases = self.optimizer.update_biases(self.biases, self.gradB)
        
    cdef na[nf64, ndim=3] im2col_for(self, na[nf64, ndim=3] im_data):
        cdef na[nf64, ndim=3] col_data = np.empty((self.im_k, self.col1_h, self.col1_w))
        for k in range(self.im_k):
            for y in range(self.col1_h):
                xc = self.st_x*(y//self.new_w)
                yc = self.st_y*(y%self.new_h)
                col_data[k,y] = im_data[k,xc:xc+self.kw,yc:yc+self.kh].flatten()
        return col_data
    
    cdef na[nf64, ndim=3] im2col_back(self, na[nf64, ndim=3] im_data):
        cdef na[nf64, ndim=3] col_data = np.empty((self.im_k, self.col1_w, self.col1_h))
        for k in range(self.im_k):
            for y in range(self.col1_w):
                xc = self.st_x*(y//self.kw)
                yc = self.st_y*(y%self.kh)
            col_data[k,y] = im_data[k,xc:xc+self.new_w,yc:yc+self.new_h].flatten()
        return col_data
    
    cdef na[nf64, ndim=3] im2col_dx(self, na[nf64, ndim=3] im_data):
        cdef na[nf64, ndim=3] col_data = np.empty((self.out_c,self.im_h*self.im_w,self.col1_w))
        for k in range(self.out_c):
            for y in range(self.im_h):
                xc = y//self.im_w
                yc = y%self.im_h
            col_data[k,y] = im_data[k,xc:xc+self.kh,yc:yc+self.kw].flatten()
        return col_data
    
    cdef na[nf64, ndim=3] col2im(self, na[nf64, ndim=2] col2_data, int out_c, int new_w, int new_h):
        cdef na[nf64, ndim=3] im_data = np.empty((out_c, new_w, new_h))
        for y in range(out_c):
            im_data[y] = col2_data[:,y].reshape(new_w,new_h)
        return im_data
    
    cdef na[nf64, ndim=3] convolute(self, na[nf64, ndim=3] col_x, na[nf64, ndim=3] filters,
                                   int out_c, int out_h, int out_w):
        out_img = np.empty((out_c, out_h, out_w))
        for k in range(out_c):
             out_img[k] = np.dot(col_x[k], filters[k])
        return out_img
    
    cdef na[nf64, ndim=3] calc_gradW(self, na[nf64, ndim=3] col_x, na[nf64, ndim= 3] delta):
        out_grad = np.empty((self.im_k,self.col1_w,self.out_c)) #self.weights.shape
        for k in range(self.im_k):
            out_grad[k] = np.dot(col_x[k], delta[0])
        return out_grad
    
    cdef na[nf64, ndim=3] delta2weight(self, na[nf64, ndim=3] delta):
        out_weight = np.empty((1,self.out_c,self.col1_h))#最後に転置
        for y in range(self.out_c):
            out_weight[0,y] = delta[y].flatten()
        return out_weight.transpose(0,2,1)
    
    cdef na[nf64, ndim=3] dilate(self, na[nf64, ndim=3] delta):
        out_delta = np.zeros((self.out_c,(self.new_h-1)*self.st_y+1,(self.new_w-1)*self.st_x+1))
        for k in range(self.out_c):
            for y in range(self.new_h):
                for x in range(self.new_w):
                    out_delta[k,y*self.st_y,x*self.st_x] = delta[k,y,x]
        return out_delta
    
    cpdef na[nf64, ndim=3] forward(self, na[nf64, ndim=3] x):
        self.x = x
        cdef output = self.col2im(np.sum(self.convolute(self.im2col_for(x),self.weights,
                            self.im_k, self.col1_h, self.out_c),axis=0),self.out_c,self.new_w,self.new_h)
        return output
    
    cpdef na[nf64, ndim=1] backward(self, na[nf64, ndim=3] delta):
        self.gradB = np.sum(delta, axis=(1,2))
        self.gradW = self.calc_gradW(self.im2col_back(self.x),self.delta2weight(delta))
        return self.col2im(np.sum(self.convolute(self.im2col_dx(np.pad(self.dilate(delta),
                        [(0,0),(self.kh,self.kh),(self.kw,self.kw)],'constant'), ), 
                    self.weights[:,::-1].transpose(2,1,0),
                self.out_c, self.im_w*self.im_h, self.im_k),axis=0),self.im_k,self.im_h,self.im_w)
    
def conv2d(kw, kh, st_x, st_y, out_c, use_bias=False):
    return "Conv2D_{},{},{},{},{},{}".format(kw,kh,st_x,st_y,out_c,use_bias)
            
                                                 
cdef class Flatten:
    cdef input_shape
    cdef public bint not_activation
    def __init__(self):
        self.not_activation = True
    cpdef list set_params(self, list input_shape):
        return [np.prod(input_shape)]
                                                 
    cpdef set_optimizer(self, optimizer):
        return
                                                 
    cpdef na[nf64, ndim=1] forward(self, na[nf64, ndim=3] x):
        self.input_shape = (x.shape[0], x.shape[1], x.shape[2])
        return x.flatten()
    
    cpdef na backward(self, na[nf64, ndim=1] delta):
        return delta.reshape(self.input_shape)

def flatten():
    return "Flatten_"
                                                 

cdef class Sigmoid:
    cdef na x, expx, sx
    cdef public bint not_activation
    def __init__(self):
        self.not_activation = False
        
    cpdef na[nf64, ndim=1] forward(self, na x):
        self.expx = self.exp(x)
        self.sx = self.expx/(1+self.expx)
        return self.sx
    
    cdef exp(self, x):
        sigmoid_range = 32
        x = np.clip(x, -sigmoid_range, sigmoid_range)
        return np.exp(x)

    cpdef na backward(self, na delta):
        return delta * self.sx/(1+self.expx)
    
def sigmoid():
    return 'Sigmoid_'
        

cdef class MSE:
    cdef na dif_sum
    cdef int batchsize_count
    cdef list input_shape
    def __init__(self):
        self.batchsize_count = 0
        
    cpdef set_params(self, list input_shape):
        self.input_shape = input_shape
        self.dif_sum = np.zeros(tuple(input_shape))
    
    cpdef nf64 forward(self, na pred, na y):
        cdef na dif = pred-y
        self.dif_sum += dif
        self.batchsize_count += 1
        return np.mean(np.square(dif))
        
    cpdef na backward(self):
        delta = self.dif_sum/self.batchsize_count
        self.dif_sum = np.zeros(tuple(self.input_shape))
        self.batchsize_count = 0
        return delta
    
def mse():
    return 'MSE'


cdef class MAE:
    cdef na dif_na
    cdef nf64 forward(self, na pred, na y):
        self.dif_na = pred-y
        return np.mean(np.abs(self.dif_na))
        
    cdef na backward(self):
        return (self.dif_na>0).astype(np.int)*2-1
    
def mae():
    return 'MAE'


cdef class CEL:
    cdef nf64 forward(self, na pred, na y):
        return -1 
    
def cel():
    return 'CEL'
