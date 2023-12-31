from pileoffeather import nn, data_loader
import numpy as np

#Define neural network model
model = nn.load(name = "test1")

#Run model
counter = 0
ones = data_loader.load(data_type = "image", color = "grayscale", folder = "ones", resize = (20,20))
for i in range(500):
    if round(model.run(ones[i])[0],3) > .5:
        counter += 1
ones = data_loader.load(data_type = "image", color = "grayscale", folder = "zeros", resize = (20,20))
for i in range(500):
    if round(model.run(ones[i])[0],3) < .5:
        counter += 1

print(str(counter/10) + "% accuracy on dataset")
