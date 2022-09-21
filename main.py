import numpy as np
import pandas as pd
from utils import *

def main():
  # load training data
  df_train = pd.read_csv('train.csv')

  # shuffle the data
  df_train = shuffle_rows(df_train)

  # split train and validation set
  train_val_split = 0.8
  train_size = round(df_train.shape[0] * train_val_split)
  data_train = df_train[:train_size,:].T
  data_val = df_train[train_size:,:].T
  
  # divide input features and target feature
  X_train = data_train[1:]
  y_train = data_train[0]
  X_val = df_train[1:]
  y_val = df_train[0]
  
  # normalize training and val sets
  X_train = normalize_pixels(X_train)
  X_val = normalize_pixels(X_val)

  # set network and optimizer parameters  
  layers_dims = [784, 256, 128, 64, 10]
  # layers_dims = [784, 10, 10]
  max_iter = 500
  alpha = 0.1

  # train the network
  params = gradient_descent_optimization(X_train, y_train, layers_dims, max_iter, alpha)
  
if __name__ == '__main__':
  main()
  