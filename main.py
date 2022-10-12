"""
Author: Riccardo Andreoni
Title: Implementation of Convolutional Neural Network from scratch.
File: main.py
"""


import numpy as np
from utils import *
import tensorflow as tf

def main():
  # Load training data
  (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
  X_train = X_train[:5000]
  y_train = y_train[:5000]

  # Define the network
  layers = [
    ConvolutionLayer(16,3), # layer with 8 3x3 filters, output (26,26,16)
    MaxPoolingLayer(2), # pooling layer 2x2, output (13,13,16)
    SoftmaxLayer(13*13*16, 10) # softmax layer with 13*13*16 input and 10 output
    ] 

  for epoch in range(4):
    print('Epoch {} ->'.format(epoch+1))
    # Shuffle training data
    permutation = np.random.permutation(len(X_train))
    X_train = X_train[permutation]
    y_train = y_train[permutation]
    # Training the CNN
    loss = 0
    accuracy = 0
    for i, (image, label) in enumerate(zip(X_train, y_train)):
      if i % 100 == 0: # Every 100 examples
        print("Step {}. For the last 100 steps: average loss {}, accuracy {}".format(i+1, loss/100, accuracy))
        loss = 0
        accuracy = 0
      loss_1, accuracy_1 = CNN_training(image, label, layers)
      loss += loss_1
      accuracy += accuracy_1
  
  
if __name__ == '__main__':
  main()
  
  