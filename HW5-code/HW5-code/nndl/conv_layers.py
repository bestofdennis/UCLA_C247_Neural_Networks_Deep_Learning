import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #

  N,C,H,W = x.shape
  F,C,HH,WW = w.shape
  x_pad = np.pad(x,((0,),(0,),(pad,),(pad,)),'constant')
  HHH = int((H - HH + 2 * pad) / stride + 1)
  WWW = int((W - WW + 2 * pad) / stride + 1)
  out = np.zeros((N,F,HHH,WWW))

  for i1 in range(N):
    for i2 in range(F):
      for i3 in range(HHH):
        for i4 in range(WWW):
          out[i1,i2,i3,i4] = np.sum(x_pad[i1,:,i3*stride:i3*stride+HH,\
          i4*stride:i4*stride+WW] * w[i2,:]) + b[i2]
          
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  dx = np.zeros(x.shape)
  dw = np.zeros(w.shape)
  db = np.zeros(b.shape)
  dxpad = np.zeros(xpad.shape)

  for i1 in range(out_height):
    for i2 in range(out_width):
      temp = xpad[:, :, i1 * stride : i1 * stride + f_height, i2 * stride : i2 * stride + f_width]
      for i3 in range(F):
        dw[i3, :, :, :] += np.sum(temp * (dout[:, i3, i1, i2])[:, None, None, None], axis = 0)
      for i4 in range(N):
        dxpad[i4, :, i1 * stride : i1 * stride + f_height, i2 * stride : i2 * stride + f_width] += np.sum(w[:, :, :, :] * (dout[i4, :, i1, i2])[:, None, None, None], axis = 0)
  dx = dxpad[:, :, pad : -pad, pad : -pad]
  db = np.sum(dout, axis = (0, 2, 3))

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #

  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  N,C,H,W = x.shape
  out_height = int((H - pool_height) / stride + 1)
  out_width = int((W - pool_width) / stride + 1)
  out = np.zeros((N, C, out_height, out_width))

  for i1 in range(N):
    for i2 in range(C):
      for i3 in range(out_height):
        for i4 in range(out_width):
          out[i1,i2,i3,i4] = np.max(x[i1,i2,i3*stride:i3*stride+pool_height,i4*stride:i4*stride+pool_width])
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  N,C,H,W = x.shape
  dx = np.zeros(x.shape)
  out_height = int((H - pool_height) / stride + 1)
  out_width = int((W - pool_width) / stride + 1)

  for i1 in range(N):
    for i2 in range(C):
      for i3 in range(out_height):
        for i4 in range(out_width):
          x_pool = x[i1,i2,i3*stride:i3*stride+pool_height,i4*stride:i4*stride+pool_width]
          max_x_pool = np.max(x_pool)
          x_mask = (x_pool == max_x_pool)
          dx[i1,i2,i3*stride:i3*stride+pool_height,i4*stride:i4*stride+pool_width] += dout[i1,i2,i3,i4] * x_mask


  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N,C,H,W = x.shape
  mode = bn_param['mode']
  eps = bn_param.get('eps',1e-5)
  momentum = bn_param.get('momentum', 0.9)
  running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))

  if mode == 'train':
    mu = (1.0 / (N * H * W) * np.sum(x, axis = (0, 2, 3))).reshape(1, C, 1, 1)
    var = (1.0 / (N * H * W) * np.sum((x - mu)**2, axis = (0, 2, 3))).reshape(1, C, 1, 1)
    xhat = (x - mu) / (np.sqrt(eps + var))
    out = gamma.reshape(1, C, 1, 1) * xhat + beta.reshape(1, C, 1, 1)
        
    running_mean = momentum * running_mean + (1.0 - momentum) * np.squeeze(mu)
    running_var = momentum * running_var + (1.0 - momentum) * np.squeeze(var)
    
    cache = (mu, var, x, xhat, gamma, beta, bn_param)
        
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
  elif mode == 'test':
    mu = running_mean.reshape(1, C, 1, 1)
    var = running_var.reshape(1, C, 1, 1)
    
    xhat = (x - mu) / (np.sqrt(eps + var))
    out = gamma.reshape(1, C, 1, 1) * xhat + beta.reshape(1, C, 1, 1)
    cache (mu, var, x, xhat, gamma, beta, bn_param)
  

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  mu, var, x, xhat, gamma, beta, bn_param = cache
  N, C, H, W = x.shape
  mode = bn_param['mode']
  eps = bn_param.get('eps',1e-5)
    
  gamma = gamma.reshape(1, C, 1, 1)
  beta = beta.reshape(1, C, 1, 1)
    
  dbeta = np.sum(dout, axis = (0, 2, 3))
  dgamma = np.sum(dout * xhat, axis = (0, 2, 3))
    
  numOt = N*H*W
  dx = (1.0 / numOt) * gamma * (var + eps)**(-1.0 / 2.0) * (numOt * dout\
         - np.sum(dout, axis = (0, 2, 3)).reshape(1, C, 1, 1)\
         - (x - mu) * (var + eps)**(-1) * np.sum(dout * (x - mu), axis = (0, 2, 3)).reshape(1, C, 1, 1))

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta
