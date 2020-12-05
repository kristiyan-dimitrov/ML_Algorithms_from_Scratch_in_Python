import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


def run_model(model,running_mode='train', train_set=None, valid_set=None, test_set=None,
    batch_size=1, learning_rate=0.01, n_epochs=1, stop_thr=1e-4, shuffle=True):
    """
    This function either trains or evaluates a model.

    training mode: the model is trained and evaluated on a validation set, if provided.
                   If no validation set is provided, the training is performed for a fixed
                   number of epochs.
                   Otherwise, the model should be evaluted on the validation set
                   at the end of each epoch and the training should be stopped based on one
                   of these two conditions (whichever happens first):
                   1. The validation loss stops improving.
                   2. The maximum number of epochs is reached.

    testing mode: the trained model is evaluated on the testing set

    Inputs:

    model: the neural network to be trained or evaluated
    running_mode: string, 'train' or 'test'
    train_set: the training dataset object generated using the class MyDataset
    valid_set: the validation dataset object generated using the class MyDataset
    test_set: the testing dataset object generated using the class MyDataset
    batch_size: number of training samples fed to the model at each training step
    learning_rate: determines the step size in moving towards a local minimum
    n_epochs: maximum number of epoch for training the model
    stop_thr: if the validation loss from one epoch to the next is less than this
              value, stop training
    shuffle: determines if the shuffle property of the DataLoader is on/off

    Outputs when running_mode == 'train':

    model: the trained model
    loss: dictionary with keys 'train' and 'valid'
          The value of each key is a list of loss values. Each loss value is the average
          of training/validation loss over one epoch.
          If the validation set is not provided just return an empty list.
    acc: dictionary with keys 'train' and 'valid'
         The value of each key is a list of accuracies (percentage of correctly classified
         samples in the dataset). Each accuracy value is the average of training/validation
         accuracies over one epoch.
         If the validation set is not provided just return an empty list.

    Outputs when running_mode == 'test':

    loss: the average loss value over the testing set.
    accuracy: percentage of correctly classified samples in the testing set.

    Summary of the operations this function should perform:
    1. Use the DataLoader class to generate training, validation, or test data loaders
    2. In the training mode:
       - define an optimizer (we use SGD in this homework)
       - call the train function (see below) for a number of epochs untill a stopping
         criterion is met
       - call the test function (see below) with the validation data loader at each epoch
         if the validation set is provided

    3. In the testing mode:
       - call the test function (see below) with the test data loader and return the results

    """

    if running_mode == 'train':

        # Variables to return
        loss = {}
        loss['train'] = []
        loss['valid'] = []

        accuracy = {}
        accuracy['train'] = []
        accuracy['valid'] = []

        train_dataloader = torch.utils.data.DataLoader(train_set
                                                        , batch_size=batch_size
                                                        , shuffle=shuffle # I think for the test dataset I don't need shuffle=True
                                                        , num_workers=2) 

        if valid_set:
            valid_dataloader = torch.utils.data.DataLoader(valid_set
                                                            , batch_size=batch_size
                                                            , shuffle=shuffle # I think for the test dataset I don't need shuffle=True
                                                            , num_workers=2) 

        # Solver necessary for training
        solver = optim.SGD(params=model.parameters() ,lr=learning_rate)

        # Training the given number of epochs
        for epoch in range(n_epochs):

            print("EPOCH ", epoch)
            model, epoch_train_loss, epoch_train_acc = _train(model, train_dataloader, solver)

            loss['train'].append(epoch_train_loss)
            accuracy['train'].append(epoch_train_acc)

            # VALIDATION
            if valid_set:

                epoch_valid_loss, epoch_valid_acc = _test(model, valid_dataloader)

                loss['valid'].append(epoch_valid_loss)
                accuracy['valid'].append(epoch_valid_acc)

                # BREAL CONDITION
                if len(accuracy['valid']) > 1:
                    new_valid_loss = loss['valid'][-1]
                    old_valid_loss = loss['valid'][-2] # Taking -2, because the new_valid_loss has already been added

                    if old_valid_loss - new_valid_loss < stop_thr:
                        break 

        return model, loss, accuracy

    ###################
    #### TEST MODE ####
    ###################

    if running_mode == 'test':

        loss = {}
        accuracy = {}

        test_dataloader = torch.utils.data.DataLoader(test_set
                                                        , batch_size=batch_size
                                                        , shuffle=shuffle # I think for the test dataset I don't need shuffle=True
                                                        , num_workers=2) 

        loss, accuracy = _test(model, test_dataloader, device)

        return loss, accuracy


def _train(model,data_loader,optimizer,device=torch.device('cpu')):

    """
    This function implements ONE EPOCH of training a neural network on a given dataset.
    Example: training the Digit_Classifier on the MNIST dataset
    Use nn.CrossEntropyLoss() for the loss function


    Inputs:
    model: the neural network to be trained
    data_loader: for loading the netowrk input and targets from the training dataset
    optimizer: the optimiztion method, e.g., SGD
    device: we run everything on CPU in this homework

    Outputs:
    model: the trained model
    train_loss: average loss value on the entire training dataset
    train_accuracy: average accuracy on the entire training dataset
    """
    minibatches = len(data_loader)

    epoch_train_loss = 0.0
    correct = 0
    total = 0  

    for batch, labels in data_loader: # Per this doc, (batch,labels) = trainloader.__getitem__(); https://pytorch.org/docs/stable/torchvision/datasets.html#cifar

        # moving batch/labels onto the gpu/cpu
        batch, labels = batch.to(device), labels.to(device)
        
        # zeroing the parameters of the model 
        # because we want to optimize them
        optimizer.zero_grad()
        
        # forward pass
        # getting the predictions from our model by passing in a mini-batch
        # the ouput will have shape (mini-batch-size, number-of-classes)
        # where each element of output is the probabliity of that example being
        # the classification correspoding to the index of the value
        output = model(batch.float()) # <---- Added .float() as a modification to avoid Runtime error per this post: https://stackoverflow.com/questions/56741087/how-to-fix-runtimeerror-expected-object-of-scalar-type-float-but-got-scalar-typ
        
        # Calculating loss
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(output, labels.long()) # <---- Added .long() per a CampusWire discussion in post #2505
        
        # Calculating accuracy
        predicted = torch.max(output.data, 1)[1]
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # backward pass
        loss.backward()
        
        # optimize the parameters
        optimizer.step()
        
        # add the loss of a mini-batch to the list of epoch loss
        epoch_train_loss += loss.item()

    # TRAINING METRICS
    train_loss = epoch_train_loss/minibatches
    train_accuracy = 100 * correct / total

    return model, train_loss, train_accuracy


def _test(model, data_loader, device=torch.device('cpu')):
    """
    This function evaluates a trained neural network on a validation set
    or a testing set.
    Use nn.CrossEntropyLoss() for the loss function

    Inputs:
    model: trained neural network
    data_loader: for loading the netowrk input and targets from the validation or testing dataset
    device: we run everything on CPU in this homework

    Output:
    test_loss: average loss value on the entire validation or testing dataset
    test_accuracy: percentage of correctly classified samples in the validation or testing dataset
    """
    correct = 0
    total = 0
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    minibatches = len(data_loader)
    epoch_loss = 0.0
    # since we're testing the model we don't need to perform backprop
    with torch.no_grad(): # Disables gradient calculation to save memory; we're sure we won't be using Tensor.backward()
        for batch, labels in data_loader:
            
            batch, labels = batch, labels # So we don't modify original dataset?
            # moving batch/labels onto the gpu/cpu
            batch, labels = batch.to(device), labels.to(device)

            output = model(batch.float()) # <---- Added .float() to solve for RuntimeError
            # this gives us the index with the highest value outputed from the last layer
            # which coressponds to the most probable label/classification for an image

            loss = loss_function(output, labels.long())
            epoch_loss += loss.item()

            predicted = torch.max(output.data, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    test_loss = epoch_loss/minibatches
    

    return test_loss, test_accuracy



