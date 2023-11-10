from pileoffeather import nn, data_loader

#Load neural network model
model = nn.load(name = "test1")

#Load example input image
input = data_loader.loadImage("example_image_one.png", (20,20), "grayscale")

#Run the neural network model
output = model.run(input)
print(output)
