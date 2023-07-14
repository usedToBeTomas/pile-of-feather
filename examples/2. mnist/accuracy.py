from pileoffeather import pof, pod
import numpy as np

#Define neural network model
model = pof.neuralNetwork(load = "mnist")

#Load mnist 10k test dataset
X = pod.load(data_type = "gz", path = "t10k-images-idx3-ubyte.gz", start_index = 16, input_number = 784, divide = 255)
Y = pod.load(data_type = "gz", path = "t10k-labels-idx1-ubyte.gz", start_index = 8)

#Accuracy tester
counter = 0
for i in range(len(X)):
    if np.argmax(model.run(X[i])) == Y[i]:
        counter +=1
print( str((counter*100)/len(Y)) + "% accuracy")
