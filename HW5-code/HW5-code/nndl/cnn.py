import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from cs231n.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

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

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #
    C,H,W = input_dim
    F = num_filters
    stride = 1
    pad = (filter_size - 1) / 2
    outh1 = (H - filter_size + 2 * pad) / stride + 1
    outw1 = (W - filter_size + 2 * pad) / stride + 1
        
    W1 = weight_scale * np.random.randn(F,C,filter_size,filter_size)
    b1 = np.zeros(F)
        
    pool_width = 2
    pool_height = 2
    pool_stride = 2
    outhPool = int((outh1 - pool_height) / pool_stride + 1)
    outwPool = int((outw1 - pool_width) / pool_stride + 1)
        
    W2 = weight_scale * np.random.randn(F*outhPool*outwPool, hidden_dim)
    b2 = np.zeros(hidden_dim)
        
    W3 = weight_scale*np.random.randn(hidden_dim,num_classes)
    b3 = np.zeros(num_classes)
    self.params.update({'W1': W1,'W2': W2,'W3': W3,'b1': b1,'b2': b2,'b3': b3})


    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #
    
    conv_layer, cache_conv_layer = conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
    N,F,HHH,WWW = conv_layer.shape
        
    input_hidden = conv_layer.reshape((N,F*HHH*WWW))
    hidden_layer, cache_hidden_layer = affine_relu_forward(input_hidden,W2,b2)
        
    N,HH = hidden_layer.shape
    scores, cache_scores = affine_forward(hidden_layer,W3,b3)
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * np.sum(W1**2)
    reg_loss += 0.5 * self.reg * np.sum(W2**2)
    reg_loss += 0.5 * self.reg * np.sum(W3**2)
    loss = data_loss + reg_loss

    grads = {}

    dx3, dW3, db3 = affine_backward(dscores,cache_scores)
    dW3 += self.reg * W3
        
    dx2, dW2, db2 = affine_relu_backward(dx3,cache_hidden_layer)
    dW2 += self.reg * W2
    
    dx2 = dx2.reshape(N,F,HHH,WWW)
    dx, dW1, db1 = conv_relu_pool_backward(dx2,cache_conv_layer)
        
    dW1 += self.reg * W1
    grads.update({'W1': dW1,'b1': db1,'W2': dW2,'b2': db2,'W3': dW3,'b3': db3})
    

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
  
  
pass
