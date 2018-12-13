import numpy as np
import h5py
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2



def zero_pad(X, pad):
    X_pad = np.pad(X, ((0,0) , (pad,pad) , (pad,pad) , (0,0)), 'constant'  )
    return X_pad


x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 2)
print ("x.shape =", x.shape)
print ("x_pad.shape =", x_pad.shape)
print ("x[1,1] =", x[1,1])
print ("x_pad[1,1] =", x_pad[1,1])

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0,:,:,0])

def conv_single_step(a_slice_prev, W, b):
    s = W * a_slice_prev
    Z = np.sum(s)
    Z = Z +b
    return Z


a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

Z = conv_single_step(a_slice_prev, W, b)
print("Z =", Z)

def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    n_H = np.int((n_H_prev - f + (2 * pad))/stride) + 1 
    n_W = np.int((n_W_prev - f + (2 * pad))/stride) + 1 
    Z = np.zeros((m, n_H , n_W , n_C))
    A_prev_pad = zero_pad(A_prev, pad)
    for i in range(m):                              
        a_prev_pad = A_prev_pad[i , : , : , :]                 
        for h in range(n_H):                           
            for w in range(n_W):                       
                for c in range(n_C):                   
                    vert_start = h*stride
                    vert_end = h*stride+ f
                    horiz_start = w*stride
                    horiz_end = w*stride + f
                    a_slice_prev = a_prev_pad[vert_start : vert_end , horiz_start : horiz_end]
                    Z[i, h, w, c] = np.sum(np.multiply(a_slice_prev ,W[:,:,:,c])) + b[0,0,0,c]
    cache = (A_prev, W, b, hparameters)
    return Z, cache
np.random.seed(1)
A_prev = np.random.randn(10,4,4,3)
W = np.random.randn(2,2,3,8)
b = np.random.randn(1,1,1,8)
hparameters = {"pad" : 2,
               "stride": 2}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
print("Z's mean =", np.mean(Z))
print("Z[3,2,1] =", Z[3,2,1])
print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
def pool_forward(A_prev, hparameters, mode = "max"):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    A = np.zeros((m, n_H, n_W, n_C))              
    for i in range(m):                  
        for h in range(n_H):            
            for w in range(n_W):        
                for c in range (n_C):   
                    vert_start = h*stride
                    vert_end = h*stride + f
                    horiz_start = w*stride 
                    horiz_end = w*stride + f
                    a_prev_slice = A_prev[i,vert_start : vert_end , horiz_start : horiz_end , c]
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    cache = (A_prev, hparameters)
    return A, cache
np.random.seed(1)
A_prev = np.random.randn(2, 4, 4, 3)
hparameters = {"stride" : 2, "f": 3}

A, cache = pool_forward(A_prev, hparameters)
print("mode = max")
print("A =", A)
print()
A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print("A =", A)

def conv_backward(dZ, cache):
    (A_prev, W, b, hparameters) = None
    (m, n_H_prev, n_W_prev, n_C_prev) = None
    (f, f, n_C_prev, n_C) = None
    stride = None
    pad = None
    (m, n_H, n_W, n_C) = None
    dA_prev = None                           
    dW = None
    db = None
    A_prev_pad = None
    dA_prev_pad = None
    
    for i in range(None):   
        a_prev_pad = None
        da_prev_pad = None
        for h in range(None):      
            for w in range(None):   
                for c in range(None): 
                    vert_start = None
                    vert_end = None
                    horiz_start = None
                    horiz_end = None
                    a_slice = None
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += None
                    dW[:,:,:,c] += None
                    db[:,:,:,c] += None
        dA_prev[i, :, :, :] = None
    return dA_prev, dW, db

np.random.seed(1)
dA, dW, db = conv_backward(Z, cache_conv)
print("dA_mean =", np.mean(dA))
print("dW_mean =", np.mean(dW))
print("db_mean =", np.mean(db))

def create_mask_from_window(x):
    mask = (x == np.max(x))
    return mask
np.random.seed(1)
x = np.random.randn(2,3)
mask = create_mask_from_window(x)
print('x = ', x)
print("mask = ", mask)

def distribute_value(dz, shape):
    (n_H, n_W) = None
    average = None
    a = None
    return a
a = distribute_value(2, (2,2))
print('distributed value =', a)

def pool_backward(dA, cache, mode = "max"):
    (A_prev, hparameters) = None
    stride = None
    f = None
    m, n_H_prev, n_W_prev, n_C_prev = None
    m, n_H, n_W, n_C = None
    dA_prev = None
    
    for i in range(None):      
        a_prev = None
        for h in range(None):              
            for w in range(None):          
                for c in range(None):      
                    vert_start = None
                    vert_end = None
                    horiz_start = None
                    horiz_end = None
                    if mode == "max":
                        a_prev_slice = None
                        mask = None
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += None
                    elif mode == "average":
                        da = None
                        shape = None
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += None
    return dA_prev
    np.random.seed(1)
A_prev = np.random.randn(5, 5, 3, 2)
hparameters = {"stride" : 1, "f": 2}
A, cache = pool_forward(A_prev, hparameters)
dA = np.random.randn(5, 4, 2, 2)

dA_prev = pool_backward(dA, cache, mode = "max")
print("mode = max")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1])  
print()
dA_prev = pool_backward(dA, cache, mode = "average")
print("mode = average")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1]) 
