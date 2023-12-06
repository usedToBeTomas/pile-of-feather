from pileoffeather import nn, data_loader
import numpy as np

#Create neural network model
model = nn.create(name = "test1", layers = [[400, 'input'], [30, 'relu'], [10, 'relu'], [1, 'sigmoid']])

#Load and prepare dataset
ones = data_loader.load(data_type = "image", color = "grayscale", folder = "ones", resize = (20,20))
zeros = data_loader.load(data_type = "image", color = "grayscale", folder = "zeros", resize = (20,20))
input = np.vstack((ones, zeros))
output = np.concatenate((np.ones(500), np.zeros(500)))

#Train the neural network and save the model
nn.mbgd(model, input, output, batch_size = 16, epoch_number = 10, rate = 0.6)
