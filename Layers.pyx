import cython
cimport cython
import numpy as np
cimport numpy as np

ctypedef np.float_t nf64
ctypedef np.ndarray na

cdef double clip_const= 709.0

cdef class ActivationLayer:
    cdef:
        public bint not_activation
        public int layer_id
    def __init__(self):
        self.not_activation = False

    
cdef class Layer:
    cdef:
        na x, gradW, gradB
        public na weights, biases
        public optimizer
        public bint not_activation
        public int layer_id
        bint use_bias
    def __init__(self):
        self.not_activation = True
        
    cpdef set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer.set_shape(self.weights, self.biases)
    
    cpdef update_parameters(self):
        self.weights = self.optimizer.update_weights(self.weights, self.gradW)
        if self.use_bias:
            self.biases = self.optimizer.update_biases(self.biases, self.gradB)


cdef class Dense(Layer):
    cdef char* init_func
    cdef int noutput
    def __init__(self, int noutput, str init_func, bint use_bias=False):
        super().__init__()
        self.noutput = noutput
        init_func_e = init_func.encode('UTF-8')
        self.init_func = init_func_e
        self.use_bias = use_bias
        
    cpdef list set_params(self, list ninput, int index):
        self.layer_id = index
        self.weights = np.random.randn(ninput[0], self.noutput)
        self.biases = np.zeros(self.noutput)
        return [self.noutput]
    
    cpdef na[nf64, ndim=1] forward(self, na[nf64, ndim=1] x):
        self.x = x
        return np.clip(np.dot(self.weights.T, x) + self.biases,-3,3)
    
    cpdef na[nf64, ndim=1] backward(self,na delta):
        self.gradW = mk_grad(delta, self.x)
        self.gradB = delta
        self.update_parameters()
        return np.dot(delta, self.weights.T)

cdef na[nf64, ndim=2] mk_grad(na delta, na x):
    mat_delta = np.matrix(delta)
    mat_x = np.matrix(x)
    return np.array(np.dot(mat_x.T, mat_delta))


cdef class Conv2D(Layer):
    cdef int kw, kh, st_x, st_y, im_k, im_w, im_h, new_w, new_h, col1_w, col1_h, out_c, h_, w_
    def __init__(self, int kw, int kh, int st_x, int st_y, int out_c, bint use_bias):
        super().__init__()
        self.kw = kw     #カーネル
        self.kh = kh
        self.h_ = int(kh/2)
        self.w_ = int(kw/2)
        self.st_x = st_x #stride
        self.st_y = st_y
        self.out_c = out_c
        self.use_bias = use_bias
 
    cpdef list set_params(self, list input_shape, int index):
        self.layer_id = index
        self.im_k, self.im_w, self.im_h = input_shape
        self.new_w = int((self.im_w - self.kw) / self.st_x) + 1
        self.new_h = int((self.im_h - self.kh) / self.st_y) + 1
        self.col1_w, self.col1_h = self.kw*self.kh, self.new_w*self.new_h
        
        self.weights = np.random.randn(self.im_k,self.col1_w,self.out_c)
        self.biases = np.zeros(self.out_c)
        return [self.out_c, self.new_w, self.new_h]

    cdef na[nf64, ndim=3] im2col_for(self, na[nf64, ndim=3] im_data):
        cdef:
            na[nf64, ndim=3] col_data
            int k,y,xc,yc
        col_data = np.empty((self.im_k, self.col1_h, self.col1_w))
        for k in range(self.im_k):
            for y in range(self.col1_h):
                xc = self.st_x*(y%self.new_w)
                yc = self.st_y*(y//self.new_h)
                col_data[k,y] = im_data[k,yc:yc+self.kh,xc:xc+self.kw].flatten()
        return col_data
    
    cdef na[nf64, ndim=3] im2col_back(self, na[nf64, ndim=3] im_data):
        cdef:
            na[nf64, ndim=3] col_data
            int k,y,xc,yc
        col_data = np.empty((self.im_k, self.col1_w, self.col1_h))
        for k in range(self.im_k):
            for y in range(self.col1_w):
                xc = self.st_x*(y%self.kw)
                yc = self.st_y*(y//self.kh)
                col_data[k,y] = im_data[k,yc:yc+self.new_h,xc:xc+self.new_w].flatten()
        return col_data

    cdef na[nf64, ndim=3] im2col_dx(self, na[nf64, ndim=3] im_data):
        cdef:
            na[nf64, ndim=3] col_data
            int k,y,xc,yc
        col_data = np.empty((self.out_c,self.im_h*self.im_w,self.col1_w))
        for k in range(self.out_c):
            for y in range(self.im_h*self.im_w):
                xc = y%self.im_w
                yc = y//self.im_h
                col_data[k,y] = im_data[k,yc:yc+self.kh,xc:xc+self.kw].flatten()
        return col_data

    cdef na[nf64, ndim=3] col2im(self, na[nf64, ndim=2] col2_data, int out_c, int new_h, int new_w):
        cdef:
            na[nf64, ndim=3] im_data 
            int y
        im_data = np.zeros((out_c, new_h, new_w))
        for y in range(out_c):
            im_data[y] = col2_data.T[y].reshape(new_h,new_w)
        return im_data

    cdef na[nf64, ndim=3] convolute(self, na[nf64, ndim=3] col_x, na[nf64, ndim=3] filters,
                                   int out_c, int out_h, int out_w):
        cdef:
            na[nf64, ndim=3] out_img
            int k
        out_img = np.empty((out_c, out_h, out_w))
        for k in range(out_c):
             out_img[k] = np.dot(col_x[k], filters[k])
        return out_img

    cdef na[nf64, ndim=3] calc_gradW(self, na[nf64, ndim=3] col_x, na[nf64, ndim=2] delta):
        cdef:
            na[nf64, ndim=3] out_grad
            int k
        out_grad = np.empty((self.im_k,self.col1_w,self.out_c)) #self.weights.shape
        for k in range(self.im_k):
            out_grad[k] = np.dot(col_x[k], delta)
        return out_grad

    cdef na[nf64, ndim=2] delta2weight(self, na[nf64, ndim=3] delta):
        cdef:
            na[nf64, ndim=2] out_weight
            int y
        out_weight = np.empty((self.out_c,self.col1_h))#最後に転置
        for y in range(self.out_c):
            out_weight[y] = delta[y].flatten()
        return out_weight.T

    cdef na[nf64, ndim=3] dilate(self, na[nf64, ndim=3] delta):
        cdef:
            na[nf64, ndim=3] out_delta
            int k,y,x
        out_delta = np.zeros((self.out_c,(self.new_h-1)*self.st_y+1,(self.new_w-1)*self.st_x+1))
        for k in range(self.out_c):
            for y in range(self.new_h):
                for x in range(self.new_w):
                    out_delta[k,y*self.st_y,x*self.st_x] = delta[k,y,x]
        return out_delta

    cpdef na[nf64, ndim=3] forward(self, na[nf64, ndim=3] x):
        self.x = x
        return np.clip(self.col2im(np.sum(self.convolute(self.im2col_for(x),self.weights,
                self.im_k, self.col1_h, self.out_c),axis=0),self.out_c,self.new_h,self.new_w)\
                    +self.biases.reshape(self.out_c,1,1),-3,3)
    
    cpdef na[nf64, ndim=1] backward(self, na[nf64, ndim=3] delta):
        self.gradB = np.sum(delta, axis=(1,2))
        self.gradW = self.calc_gradW(self.im2col_back(self.x),self.delta2weight(delta))
        self.update_parameters()
        return self.col2im(np.sum(self.convolute(self.im2col_dx(np.pad(self.dilate(delta),
                    [(0,0),(self.kh,self.kh),(self.kw,self.kw)],'constant')), 
                self.weights[:,::-1,:].transpose(2,1,0),self.out_c,self.im_w*self.im_h,self.im_k),
            axis=0),self.im_k,self.im_h,self.im_w)


cdef class Padding2D(ActivationLayer):
    cdef:
        list input_shape, output_shape
        int pad
    def __init__(self, int pad):
        super().__init__()
        self.pad = pad

    cpdef list set_params(self, list input_shape, int index):
        self.layer_id = index
        self.input_shape = input_shape
        self.output_shape = [input_shape[0]]
        self.output_shape += [self.pad*2+x for x in input_shape[1:]]
        return self.output_shape

    cpdef na[nf64, ndim=3] forward(self, na[nf64, ndim=3] x):
        return np.pad(x, [(0,0),(self.pad,self.pad),(self.pad,self.pad)],'constant')
    
    cpdef na[nf64, ndim=3] backward(self, na[nf64, ndim=3] delta):
        return delta[:,self.pad:self.output_shape[1]-self.pad
                     ,self.pad:self.output_shape[2]-self.pad]


cdef class MaxPooling2D(ActivationLayer):
    cdef:
        list input_shape, output_shape, col_shape
        int kernel, nc
        na argbuffer
        bint bottom_pad, right_pad
    def __init__(self, int kernel):
        super().__init__()
        self.kernel = kernel

    cpdef list set_params(self, list input_shape, int index):
        self.layer_id = index
        self.nc = input_shape[0]
        self.input_shape = input_shape
        self.output_shape = [self.nc]
        self.output_shape += [int(np.ceil(x/self.kernel)) for x in input_shape[1:]]
        self.col_shape = [self.nc,int(np.prod(np.array(self.output_shape[1:]))),self.kernel**2]
        self.bottom_pad = False if self.output_shape[1] == input_shape[1]/self.kernel else True
        self.right_pad = False if self.output_shape[2] == input_shape[2]/self.kernel else True
        return self.output_shape
    
    cpdef na[nf64, ndim=3] forward(self, na[nf64, ndim=3] input_img):
        cdef na[nf64, ndim=3] input_col
        input_col = self.im2col(input_img)
        self.argbuffer = np.argmax(input_col,axis=2)
        return self.col2im(np.max(input_col,axis=2))
    
    cpdef na[nf64, ndim=3] backward(self, na[nf64, ndim=3] delta):
        cdef:
            na[nf64, ndim=3] out_delta
            int k,y,x,yc,xc
        out_delta = np.zeros((self.input_shape))
        for k in range(self.nc):
            for y in range(self.output_shape[1]):
                for x in range(self.output_shape[2]):
                    yc = self.argbuffer[k,y*x]//self.kernel
                    xc = self.argbuffer[k,y*x]%self.kernel
                    out_delta[k,yc,xc] = delta[k,y,x]
        return out_delta
    
    cdef na[nf64, ndim=3] im2col(self, na[nf64, ndim=3] img):
        cdef:
            na[nf64, ndim=3] col
            int k,y,xc,yc
        if self.bottom_pad:
            img = np.pad(img,[(0,0),(0,self.kernel),(0,0)],'constant')
        if self.right_pad:
            img = np.pad(img,[(0,0),(0,0),(0,self.kernel)],'constant')
        col = np.zeros((self.col_shape))
        for k in range(self.nc):
            for y in range(self.col_shape[1]):
                yc = self.kernel*(y//self.output_shape[1])
                xc = self.kernel*(y%self.output_shape[2])
                col[k,y] = img[k,yc:yc+self.kernel,xc:xc+self.kernel].flatten()
        return col
    
    cdef na[nf64, ndim=3] col2im(self, na[nf64, ndim=2] col):
        cdef na[nf64, ndim=3] out_img
        out_img = np.zeros((self.output_shape))
        for k in range(self.nc):
            out_img[k] = col[k].reshape(*self.output_shape[1:])
        return out_img
        

cdef class Flatten(ActivationLayer):
    cdef input_shape       
    def __init__(self):
        super().__init__()
        
    cpdef list set_params(self, list input_shape, index):
        self.layer_id = index
        return [np.prod(np.array(input_shape))]
    
    cpdef na[nf64, ndim=1] forward(self, na x):
        self.input_shape = (x.shape[0], x.shape[1], x.shape[2])
        return x.flatten()
    
    cpdef na backward(self, na[nf64, ndim=1] delta):
        return delta.reshape(self.input_shape)


cdef class Sigmoid(ActivationLayer):
    cdef na x, expx, sx
    def __init__(self):
        super().__init__()
        
    cpdef list set_params(self, list input_shape, index):
        self.layer_id = index
        return input_shape
    
    cpdef na[nf64, ndim=1] forward(self, na x):
        self.expx = self.exp(x)
        self.sx = self.expx/(1+self.expx)
        return self.sx

    cpdef na backward(self, na delta):
        return delta * self.sx/(1+self.expx)
        
    cdef exp(self, x):
        sigmoid_range = 32
        x = np.clip(x, -sigmoid_range, sigmoid_range)
        return np.exp(x)


cdef class LeakyReLU(ActivationLayer):
    cdef:
        na tfbuffer
        int slope
    
    def __init__(self, int slope):
        super().__init__()
        self.slope = slope
        
    cpdef list set_params(self, list input_shape, index):
        self.layer_id = index
        return input_shape
    
    cpdef na forward(self, na x):
        self.tfbuffer = x<0
        x[self.tfbuffer] = x[self.tfbuffer]*self.slope
        return x
    
    cpdef na backward(self, na delta):
        delta[self.tfbuffer] = delta[self.tfbuffer]*self.slope
        return delta
        
