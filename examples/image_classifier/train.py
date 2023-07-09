from pileoffeather import pof, pod
import numpy as np

#Define neural network model
model = pof.neuralNetwork(layers = [[400,"input"],[30,"relu"],[10,"relu"],[1,"sigmoid"]], name = "test1")

#Load the image dataset, 500 ones images and 500 zeros images
ones = pod.load(data_type = "image", folder = "ones", resize = (20,20))
zeros = pod.load(data_type = "image", folder = "zeros", resize = (20,20))
input = np.vstack((ones, zeros))#Stack them

#Generate expected output, first 500 images should output 1, the other 500 0
output = np.concatenate((np.ones(500), np.zeros(500)))

#Train the neural network using backpropagation, it automatically saves the weights and biases at the end of the 100 epochs
model.train(input, output, batch_size = 16, epoch_number = 100, rate = 0.6)
