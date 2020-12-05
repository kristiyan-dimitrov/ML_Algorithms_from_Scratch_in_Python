import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class Text_Classifier(nn.Module):
    """
    This is the class that creates a neural network for classifying handwritten digits
    from the MNIST dataset.

    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size specified when creating the class
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (number based on size specified when creating the class)

    Activation function: ReLU for both hidden layers

    """
    def __init__(self, input_size, output_size):
        super(Text_Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256) # <-------- Important Modification
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size) # <-------- Important Modification


    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# FOR GLOVE REPRESENTATION
class Text_Classifier_Glove(nn.Module):
    def __init__(self, input_size, output_size):
        super(Text_Classifier_Glove, self).__init__()
        self.fc1 = nn.Linear(input_size, 128) # <-------- Important Modification
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size) # <-------- Important Modification


    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x