"""
Author: Riccardo Andreoni
Title: Implementation of Convolutional Neural Network from scratch.
File: utils.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Convolution:
    def __init__(self, kernel_num, kernel_size):
        """
        Constructor takes as input the number of kernels and their size. I assume only squared filters of size kernel_size x kernel_size
        """
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        # Generate random filters of shape (kernel_num, kernel_size, kernel_size). Divide by kernel_size^2 for weight normalization
        self.kernels = np.random.randn(kernel_num, kernel_size, kernel_size) / (kernel_size**2)

    def patches_generator(self, image):
        """
        Divide the input image in patches to be used during convolution.
        Yields the tuples containing the patches and their coordinates.
        """
        # Extract image height and width
        image_h, image_w = image.shape
        self.image = image
        # The number of patches, given a fxf filter is h-f+1 for height and w-f+1 for width
        for h in range(image_h-self.kernel_size+1):
            for w in range(image_w-self.kernel_size+1):
                patch = image[h:h+self.kernel_size, w:w+self.kernel_size]
                yield (patch, h, w)
                #usare yield perchè è un generator, che genera i patches e le loro coordinate
    
    def forward_prop(self, image):
        # Extract image height and width
        image_h, image_w = image.shape
        # Initialize the convolution output volume of the correct size
        convolution_output = np.zeros((image_h-self.kernel_size+1, image_w-self.kernel_size+1, self.kernel_num))
        # Unpack the generator
        for (patch, h, w) in self.patches_generator(image):
            # Perform convolution for each patch
            convolution_output[h,w] = np.sum(patch*self.kernels, axis=(1,2))
        return convolution_output
    
    def back_prop(self, dE_dY, alpha):
        """
        Takes the gradient of the loss function with respect to the output and computes the gradients of the loss function with respect
        to the kernels' weights.
        dE_dY comes from the following layer, typically max pooling layer.
        It updates the kernels' weights
        """
        # Initialize gradient of the loss function with respect to the kernel weights
        dE_dk = np.zeros(self.kernels.shape)
        for (patch, h, w) in self.patches_generator(self.image):
            for f in range(self.kernel_num):
                dE_dk[f] += patch * dE_dY[h, w, f]
        # Update the parameters
        self.kernels -= alpha*dE_dk
        return dE_dk



# Test
df_train = pd.read_csv('train.csv')
img = df_train.iloc[40,:].values[1:]
img = np.reshape(img,(28,28))
plt.imshow(img, cmap='gray')
plt.show()
print(img.shape)

# Test with a convolution of 16 filters of size 3x3
my_conv = Convolution(16,3)
output = my_conv.forward_prop(img)
output.shape

# Plot 16th volume after the convolution
plt.imshow(output[:,:,15], cmap='gray')
plt.show()