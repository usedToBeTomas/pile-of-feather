<div align="center">
<h1>POF - pile of feather</h1>
<img src="https://github.com/usedToBeTomas/pile-of-feather/blob/main/images/pof2.png" width="400" height="400" />

Pileoffeather is not an alternative to big ml library like pytorch or tensorflow, it lacks features and optimization, such as gpu support. The goal is to create a lightweight library of about 100 lines of code that is easy to use and quick to implement for creating small projects or experiment with ml.

<h3>

[Documentation](https://github.com/usedToBeTomas/pile-of-feather#documentation) | [Examples](https://github.com/usedToBeTomas/pile-of-feather#examples)

</h3>

</div>

Install module
```cmd
pip install pileoffeather
```
# Examples
## examples/image_classifier
Handwritten digit image classifier, dataset is made out of 500 images of ones and 500 images of zeros taken from the mnist dataset. The first code snippet is defining the neural network model, uploading the dataset and than training the model
```python
from pileoffeather import nn, data_loader
import numpy as np

#Create neural network model
model = nn.create(name = "test1", layers = [[400, 'input'], [30, 'relu'], [10, 'relu'], [1, 'sigmoid']])

#Load and prepare dataset
ones = data_loader.load(data_type = "image", color = "grayscale", folder = "ones", resize = (20,20))
zeros = data_loader.load(data_type = "image", color = "grayscale", folder = "zeros", resize = (20,20))
input = np.vstack((ones, zeros))
output = np.concatenate((np.ones(500), np.zeros(500)))

#Train the neural network and save the model
nn.backpropagation(model, input, output, batch_size = 16, epoch_number = 10, rate = 0.6)
```
This second code snippet is used to run the trained model
```python
from pileoffeather import nn, data_loader

#Load neural network model
model = nn.load(name = "test1")

#Load example input image
input = data_loader.loadImage("example_image_one.png", (20,20), "grayscale")

#Run the neural network model
output = model.run(input)
print(output)
```

---

# Documentation
The module consists of 3 files: nn.py (create and train nn), data_loader.py (for loading and converting data) and engine.py (Hidden core neural network engine < 100 lines of code).
## nn.py - neural network module

```python
#Import module
from pileoffeather import nn

#Define neural network model, the available activation functions are "sigmoid","relu","leakyRelu"
model = nn.create(name = "test1", layers = [[400,""],[50,"relu"],[10,"relu"],[1,"sigmoid"]])

#Load an exsisting model
model = nn.load(name = "test1")

#Complete backpropagation over all the dataset, uses model.computeBatch in a loop
nn.backpropagation(model, input_matrix, output_matrix, batch_size = 16, epoch_number = 100, rate = 0.03)

#Save the model
model.save()

#Use the neural network
output = model.run(input)

#Compute single mini_batch pass, model.computeBatch(batch_input, batch_output, batch_size, learning_rate)
model.computeBatch(batch_input, batch_output, 16, 0.3)

#Pop a layer out of the neural network model (ex. remove layer at index 2)
model.pop(2)

#Insert a model layers over an other model, training is preserved (ex. at index 2)
model.insert(model1,2)
```

## data_loader.py - data load module

```python
#Import module
from pileoffeather import data_loader

#Load dataset of images from local folder
dataset = data_loader.load(data_type = "image", color = "grayscale", folder = "folder_name_containing_all_images", resize = (20,20))

#Load training input data of mnist, normalize input from 0 to 1 using divide = 255
dataset = data_loader.load(data_type = "gz", path = "train-images-idx3-ubyte.gz", start_index = 16, input_number = 784, divide = 255)

#Load training output data of mnist, use one_hot encoding to convert a decimal number to an array
#4 -> [0,0,0,0,1,0,0,0,0,0] 0 -> [1,0,0,0,0,0,0,0,0,0]
dataset = data_loader.load(data_type = "gz", path = "train-labels-idx1-ubyte.gz", start_index = 8, one_hot = 10)

#Load a single image to feed the neural network loadImage(name, resize, color)
input = data_loader.loadImage("example_image.png", (20,20), "grayscale")

#Convert neural network output to image and save, saveImage(neural_network_output, image_path, resize, color)
data_loader.saveImage(neural_network_output, "image_path_and_name", (20,20), "grayscale")
```

---

### TODO
- Improve pod (pileofdata) with better syntax and more data loading functions
- Add recurrent neural networks and common architectures like transformer or gan
- Implement a training method for those architectures
- Possibility to live graph loss, accuracy or similar stats during training
