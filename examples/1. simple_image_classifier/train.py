from pileoffeather import pof, pod
import numpy as np

#Define neural network model
model = pof.neuralNetwork(layers = [[400,"input"],[30,"relu"],[10,"relu"],[1,"sigmoid"]], name = "test1")

#Load the images for the dataset, 500 ones images and 500 zeros images
ones = pod.load(data_type = "image", color = "grayscale", folder = "ones", resize = (20,20))
zeros = pod.load(data_type = "image", color = "grayscale", folder = "zeros", resize = (20,20))
input = pod.merge(ones, zeros)

#Generate expected output, first 500 images should output 1, the other 500 0
output = np.concatenate((np.ones(500), np.zeros(500)))

#Train the neural network using backpropagation
pof.train(model, input, output, batch_size = 16, epoch_number = 100, rate = 0.6)
