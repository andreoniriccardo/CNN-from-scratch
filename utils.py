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

def forward_prop(X, params):
  """
  Forward propagation for the L layers.
  First (L-1) layers: relu activation
  Last layer: softmax activation
  """
  # number of layers (note: params contains W and b for each layer, so it's necessary to do //2)
  L = len(params) // 2
  
  activations = {}
  activations['A0'] = X

  # for layers 1 to L-1 apply relu activation
  for l in range(1,L):
    activations['Z'+str(l)] = np.dot(params['W'+str(l)], activations['A'+str(l-1)]) + params['b'+str(l)]
    activations['A'+str(l)] = relu(activations['Z'+str(l)])

  # for layer L apply softmax activation
  activations['Z'+str(L)] = np.dot(params['W'+str(L)], activations['A'+str(L-1)]) + params['b'+str(L)]
  activations['A'+str(L)] = softmax(activations['Z'+str(L)])  
  
  return activations


def back_prop(activations, params, Y):
  """
  Inputs:
  activations: dictionary like {'A0':..., 'A1':..., 'Z1':..., 'A2':..., ...}
  params: dictionary like {'W1':..., 'b1':..., 'W2':...}
  Y
  Output:
  grads: dictionary like {'dW1':..., 'db1':..., ...}
  """
  
  L = len(params) // 2  
  one_hot_Y = one_hot(Y)  
  m = one_hot_Y.shape[1]
  
  derivatives = {}
  grads = {}
  
  # for layer L
  derivatives['dZ'+str(L)] = (activations['A'+str(L)] - one_hot_Y)
  grads['dW'+str(L)] = 1 / m * np.dot(derivatives['dZ'+str(L)], activations['A'+str(L-1)].T)
  grads['db'+str(L)] = 1 / m * np.sum(derivatives['dZ'+str(L)])
   
  # for layers L-1 to 1
  for l in reversed(range(1, L)):
    derivatives['dZ'+str(l)] = np.dot(params['W'+str(l+1)].T, derivatives['dZ'+str(l+1)]) * deriv_relu(activations['Z'+str(l)])
    grads['dW'+str(l)] = 1 / m * np.dot(derivatives['dZ'+str(l)], activations['A'+str(l-1)].T)
    grads['db'+str(l)] = 1 / m * np.sum(derivatives['dZ'+str(l)], axis=1, keepdims=True)
  
  return grads

def update_params(params, grads, alpha):
  # number of layers
  L = len(params) // 2

  params_updated = {}
  for l in range(1, L+1):
    params_updated['W'+str(l)] = params['W'+str(l)] - alpha*grads['dW'+str(l)]
    params_updated['b'+str(l)] = params['b'+str(l)] - alpha*grads['db'+str(l)]

  return params_updated

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

def gradient_descent_optimization(X, Y, layers_size, max_iter, alpha):
  # initiallize parameters Wl, bl for layers l=1,...,L
  params = init_params(layers_size)
  L = len(params)//2
  accuracies = []
  losses = []
  for iter in range(1,max_iter+1):
    # compute activations: forward propagation
    activations = forward_prop(X, params)
    # make prediction
    Y_hat = get_predictions(activations['A'+str(L)])
    # compute accuracy
    accuracy = get_accuracy(Y_hat, Y)
    accuracies.append(accuracy)
    
    # compute loss (cross_entropy)
    loss = cross_entropy(one_hot(Y), activations['A'+str(L)])
    losses.append(loss)

    # compute gradients: back propagation
    grads = back_prop(activations, params, Y)

    # update the parameters
    params = update_params(params, grads, alpha)

    if iter % 10 == 0:
      print('Accuracy at iter {}: {}'.format(iter, accuracy))

  # plot training accuracy and loss
  plt.plot(range(1, max_iter+1), accuracies, '-', color=sns.color_palette('deep')[0], linewidth=2, label='Training Accuracy')
  plt.plot(range(1, max_iter+1), losses, ':', color=sns.color_palette('deep')[2], linewidth=2, label='Training Loss')
  plt.title("Network's Architecture: {}".format(layers_size))
  plt.legend(loc="upper right")
  plt.xlabel("X axis label")
  plt.savefig('images/training_acc_loss_{}.png'.format(layers_size), format='png', dpi=1200)
    
  return params