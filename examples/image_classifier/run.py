from pileoffeather import pof, pod
import numpy as np

#Define neural network model
model = pof.neuralNetwork(load = "test1")

#Run model
output = model.run(pod.loadImage("example_image_zero.png", (20,20)))

#Print result
print(output)
