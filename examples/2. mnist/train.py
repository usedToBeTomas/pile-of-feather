from pileoffeather import pof, pod

#Define neural network model
model = pof.neuralNetwork(layers = [[784,""],[128,"relu"],[10,"sigmoid"]], name = "mnist")

#Upload mnist dataset
input_dataset = pod.load(data_type = "gz", path = "train-images-idx3-ubyte.gz", start_index = 16, input_number = 784, divide = 255)
output_dataset = pod.load(data_type = "gz", path = "train-labels-idx1-ubyte.gz", start_index = 8, one_hot = 10)

#Train the neural network using backpropagation
pof.train(model, input_dataset, output_dataset, batch_size = 12, epoch_number = 2, rate = 1)
