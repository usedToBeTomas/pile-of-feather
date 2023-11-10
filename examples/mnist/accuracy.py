from pileoffeather import nn, data_loader
import numpy as np

#Define neural network model
model = nn.load(name = "mnist")

#Load mnist 10k test dataset
X = data_loader.load(data_type = "gz", path = "t10k-images-idx3-ubyte.gz", start_index = 16, input_number = 784, divide = 255)
Y = data_loader.load(data_type = "gz", path = "t10k-labels-idx1-ubyte.gz", start_index = 8)

#Accuracy tester
counter = 0
for i in range(len(X)):
    if np.argmax(model.run(X[i])) == Y[i]:
        counter +=1
print(str((counter*100)/len(Y)) + "% accuracy")
