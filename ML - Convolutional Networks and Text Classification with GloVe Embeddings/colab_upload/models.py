import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Please read the free response questions before starting to code.


class Dog_Classifier_Conv(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying dog breeds
    from the DogSet dataset.

    Network architecture (see problems.md for more information):
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    There should be a maxpool after each convolution.

    The sequence of operations looks like this:

        1. Apply convolutional layer with stride and kernel size specified
            - note: uses hard-coded in_channels and out_channels
            - read the problems to figure out what these should be!
        2. Apply the activation function (ReLU)
        3. Apply 2D max pooling with a kernel size of 2

    Inputs:
    kernel_size: list of length 2 containing kernel sizes for the two convolutional layers
                 e.g., kernel_size = [(3,3), (3,3)]
    stride: list of length 2 containing strides for the two convolutional layers
            e.g., stride = [(1,1), (1,1)]

    """

    def __init__(self, kernel_size, stride):
        super(Dog_Classifier_Conv, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size[0], stride[0]) # First Convolutional layer - should have 3 input channels, because DogSet images are RGB
        self.pool = nn.MaxPool2d(2, 2) # Maxpooling
        self.conv2 = nn.Conv2d(16, 32, kernel_size[1], stride[1]) # 2nd Conv. layer - should have 16 input layers, because that's
        self.pool = nn.MaxPool2d(2, 2) # Maxpooling

        # (I-K)/S + 1 => Size of channel after going through convolution with kernel size K, stride S, input is I
        # The formula is actually the same for a Maxpool when there is no padding or dilation i.e. with defaults

        size_first_conv = (64-kernel_size[0][0])/stride[0][0]+1 # Assuming square kernels & strides; images in DogSet are 64x64
        size_first_pool = (size_first_conv-2)/2+1
        size_second_conv = (size_first_pool-kernel_size[1][0])/stride[1][0]+1
        size_second_pool = (size_second_conv-2)/2+1
        input_for_final_layer = 32 * size_second_pool * size_second_pool # 32 for the number of channels from last conv. layer

        self.fc3 = nn.Linear(int(input_for_final_layer), 10)


    def forward(self, inputs):
        # Note that the ordering of dimensions in the input may not be what you
        # need for the convolutional layers.  The permute() function can help.
        x = self.pool(F.relu(self.conv1(inputs.permute(0,3,1,2)))) # Conv. Layer expects input in shape [N, C, H, W], but default is [N, H, W, C]
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x)) # Flattens
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Synth_Classifier(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying
    synthesized images.

    Network architecture (see problems.md for more information):
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Third hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 2)

    Activation function: ReLU for both hidden layers

    There should be a maxpool after each convolution.

    The sequence of operations looks like this:

        1. Apply convolutional layer with stride and kernel size specified
            - note: uses hard-coded in_channels and out_channels
            - read the problems to figure out what these should be!
        2. Apply the activation function (ReLU)
        3. Apply 2D max pooling with a kernel size of 2

    Inputs:
    kernel_size: list of length 3 containing kernel sizes for the three convolutional layers
                 e.g., kernel_size = [(5,5), (3,3),(3,3)]
    stride: list of length 3 containing strides for the three convolutional layers
            e.g., stride = [(1,1), (1,1),(1,1)]

    """

    def __init__(self, kernel_size, stride):
        super(Synth_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size[0], stride[0]) 
        self.pool = nn.MaxPool2d(2, 2) # Maxpooling
        self.conv2 = nn.Conv2d(2, 4, kernel_size[1], stride[1]) # 2nd Conv. layer - should have 16 input layers, because that's
        self.pool = nn.MaxPool2d(2, 2) # Maxpooling
        self.conv3 = nn.Conv2d(4, 8, kernel_size[2], stride[2]) # 2nd Conv. layer - should have 16 input layers, because that's
        self.pool = nn.MaxPool2d(2, 2) # Maxpooling

        size_first_conv = np.floor((28-kernel_size[0][0])/stride[0][0]+1) # Assuming square kernels & strides; images in SynthData are 28x28, single channel
        size_first_pool = np.floor((size_first_conv-2)/2+1)
        size_second_conv = np.floor((size_first_pool-kernel_size[1][0])/stride[1][0]+1)
        size_second_pool = np.floor((size_second_conv-2)/2+1)
        size_third_conv = np.floor((size_second_pool-kernel_size[2][0])/stride[2][0]+1)
        size_third_pool = np.floor((size_third_conv-2)/2+1)

        input_for_final_layer = 8 * size_third_pool * size_third_pool # 8 for the number of channels from last conv. layer
        self.fc4 = nn.Linear(int(input_for_final_layer), 2)

    def forward(self, inputs):
        # Note that the ordering of dimensions in the input may not be what you
        # need for the convolutional layers.  The permute() function can help.
        x = self.pool(F.relu(self.conv1(inputs.permute(0,3,1,2)))) # Conv. Layer expects input in shape [N, C, H, W], but default is [N, H, W, C]
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x)) # Flattens
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

