from pileoffeather import pof, pod
import numpy as np

#Define neural network model
model = pof.neuralNetwork(load = "test1")

#Run model
input = pod.loadImage("example_image_one.png", (20,20), "grayscale")
output = model.run(input)

#Print result
print(round(output[0],3))
