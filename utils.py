"""
Author: Riccardo Andreoni
Title: Implementation of Convolutional Neural Network from scratch.
File: utils.py
"""


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns

def shuffle_rows(data):
  """
  This function shuffles the row of the input dataframe.
  Input: data (pandas.DataFrame or ndarra)
  Output: shuffled data (ndarray)
  """
  # Convert input dataframe to ndarray
  data = np.array(data)
  np.random.shuffle(data)
  return data

def normalize_pixels(data):
  return data/255.


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
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        # vedi formula su notebook 'Convolutional Neural Networks assignment_01' con padding=0, stride=1
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernel_shape = (depth, input_depth, kernel_size, kernel_size)
        # generate random initial values for kernels and biases
        self.kernels = np.random.randn(*self.kernel_shape)
        self.biases = np.random.randn(*self.output_shape)
    
    def forward(self, input):
        self.input = input
        # initialize the output starting from the biases (they have the same shape as the output)
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i,j], 'valid')
        return self.output






def valid_cross_corr():
    












"""
parte vecchia
"""


def init_params(layers_dims):
  params = {}
  for layer in range(1,len(layers_dims)):
    params['W'+str(layer)] = np.random.randn(layers_dims[layer], layers_dims[layer-1]) * np.sqrt(1. / layers_dims[layer])
    params['b'+str(layer)] = np.random.randn(layers_dims[layer],1) * np.sqrt(1. / layers_dims[layer])
    
  return params
  
def relu(Z):
  return np.maximum(Z,0)

def softmax(Z):
  A = np.exp(Z) / sum(np.exp(Z))
  return A

def deriv_relu(Z):
  return Z > 0

def deriv_softmax(Z):    
    dZ = np.exp(Z) / sum(np.exp(Z)) * (1. - np.exp(Z) / sum(np.exp(Z)))
    return dZ

def one_hot(Y):
  """
  Y should have shape n,1 where n is the number of classes.
  Y comes in integer form (e.g. 4) and should be converted in binary shape:
  Y = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]^T
  """
  # create temporary zeros array of shape (m,n), where m is the number
  # of training examples in Y, n is the number of classes in Y
  Y_one_hot = np.zeros((Y.shape[0], Y.max()+1))
  # set to 1 the corret indices
  Y_one_hot[np.arange(Y.shape[0]), Y] = 1
  # transpose
  Y_one_hot = Y_one_hot.T
  return Y_one_hot

def conv_forward_prop(X, params):
  """
  Forward propagation for the L layers.
  First (L-1) layers: relu activation
  Last layer: softmax activation
  """
  # Retrieve dimensions from input
  
  
  
  return activations


def back_prop(activations, params, Y):
  """
  Inputs:
  
  Output:
  
  """
  
  
  return grads

def update_params(params, grads, alpha):
  

  return 

def cross_entropy(Y_one_hot, Y_hat, epsilon=1e-12):
  """
  Compute cross entropy between target Y_one_hot (encoded as one-hot vector)
  and predictions Y_hat.
  Inputs: Y_one_hot (k, m) ndarray
          Y_hat (k, m) ndarray
          k: number of classes
          N: number of samples
  Output: cross entropy (scalar)
  sources:
    code: https://stackoverflow.com/questions/47377222/what-is-the-problem-with-my-implementation-of-the-cross-entropy-function
    formula: https://medium.com/unpackai/cross-entropy-loss-in-ml-d9f22fc11fe0#:~:text=Cross%2Dentropy%20can%20be%20calculated,*%20log(Q(x))
  """
  
  # clip predictions to avoid values of 0 and 1
  Y_hat = np.clip(Y_hat, epsilon, 1.-epsilon)
  # sum on the columns of Y_hat * np.log(Y), then take the mean 
  # between the m samples
  cross_entropy = -np.mean(np.sum(Y_one_hot * np.log(Y_hat), axis=0))
  return cross_entropy

def get_predictions(AL):
  # get the max index by the columns  
  return np.argmax(AL, axis=0)
  
def get_accuracy(Y_hat, Y):
  """
  Given the predicted classes Y_hat and the actual classes Y, returns the accuracy of the prediction
  Input:
  Y_hat (1,m) ndarray
  Y (1,m) ndarray
  Output:
  accuracy (scalar)
  """
  return np.sum(Y_hat == Y) / Y.size

  # plot training accuracy and loss
  plt.plot(range(1, max_iter+1), accuracies, '-', color=sns.color_palette('deep')[0], linewidth=2, label='Training Accuracy')
  plt.plot(range(1, max_iter+1), losses, ':', color=sns.color_palette('deep')[2], linewidth=2, label='Training Loss')
  plt.title("Network's Architecture: {}".format(layers_size))
  plt.legend(loc="upper right")
  plt.xlabel("X axis label")
  plt.savefig('images/training_acc_loss_{}.png'.format(layers_size), format='png', dpi=1200)
    
  return params