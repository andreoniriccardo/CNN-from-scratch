"""
Author: Riccardo Andreoni
Title: Implementation of Convolutional Neural Network from scratch.
File: utils.py
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        
        pass
    
    def backward(self, output_gradient, learning_rate):
        
        
        pass

# create Convolutional_Layer class. It inherits from Layer class
class Convolutional_Layer(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        # unpack input shape
        input_depth, input_height, input_width = input_shape
        
        # initialize layers' attributes
        self.depth = depth # number of kernels in the conv layer e.g. 2
        
        self.input_shape = input_shape # e.g. (3,3,3)
        self.input_depth = input_depth # e.g. 3
        
        # vedi formula su notebook 'Convolutional Neural Networks assignment_01' con padding=0, stride=1
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1) # e.g. (2,2,2)
        self.kernel_shape = (depth, input_depth, kernel_size, kernel_size) # e.g. (2,3,2,2)
        
        # generate random initial values for kernels and biases
        self.kernels = np.random.randn(*self.kernel_shape) # e.g. (2,3,2,2)
        self.biases = np.random.randn(*self.output_shape) # e.g. (2,2,2)
    
    def forward(self, input):
        self.input = input
        
        # initialize the output starting from the biases (they have the same shape as the output)
        self.output = np.copy(self.biases) # e.g. (2,2,2)
        
        for i in range(self.depth): # e.g. 0,1
            for j in range(self.input_depth): # e.g. 0,1,2
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i,j], 'valid') # correlate2d is not commutative
        return self.output





