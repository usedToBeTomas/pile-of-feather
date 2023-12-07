from pileoffeather import nn, data_loader

#Define neural network model
model = nn.create(name = "mnist", layers = [[784, "input"], [144, 'relu'], [10, 'sigmoid']])

#Upload mnist dataset
X = data_loader.load(data_type = "gz", path = "train-images-idx3-ubyte.gz", start_index = 16, input_number = 784, divide = 255)
Y = data_loader.load(data_type = "gz", path = "train-labels-idx1-ubyte.gz", start_index = 8, one_hot = 10)

#Train the neural network using backpropagation
nn.mbgd(model, X, Y, batch_size = 16, epoch_number = 2, rate = 1)
