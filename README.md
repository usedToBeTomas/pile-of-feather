<div align="center">
<h1>POF - pile of feather</h1>
<img src="https://github.com/usedToBeTomas/pile-of-feather/blob/main/images/pof.png" width="400" height="400" />

This library is not an alternative to big ml library like pytorch or tensorflow, it lacks features and optimization, such as gpu support. The goal is to create a lightweight library of about 100 lines of code that is easy to use and quick to implement for creating small projects or experiments. The library is split into 2 files, pof.py (for creating and using nn), pod.py (for loading and converting data).

<h3>

[Documentation](https://github.com/usedToBeTomas/pile-of-feather#documentation) | [Examples](https://github.com/usedToBeTomas/pile-of-feather#examples)

</h3>

</div>

Install module
```cmd
pip install pileoffeather
```
# Examples
## 1. examples/image_classifier
Handwritten digit image classifier, dataset is made out of 500 images of ones and 500 images of zeros taken from the mnist dataset. The first code snippet is defining the neural network model, uploading the dataset and than training the model
```python
from pileoffeather import pof, pod
import numpy as np

#Define neural network model
model = pof.neuralNetwork(layers = [[400,""],[30,"relu"],[10,"relu"],[1,"sigmoid"]], name = "test1")

#Load the images for the dataset, 500 ones images and 500 zeros images
ones = pod.load(data_type = "image", color = "grayscale", folder = "ones", resize = (20,20))
zeros = pod.load(data_type = "image", color = "grayscale", folder = "zeros", resize = (20,20))
input = pod.merge(ones, zeros)

#Generate expected output, first 500 images should output 1, the other 500 0
output = np.concatenate((np.ones(500), np.zeros(500)))

#Train the neural network using backpropagation
pof.train(model, input, output, batch_size = 16, epoch_number = 100, rate = 0.6)
```
Run the trained model
```python
from pileoffeather import pof, pod

#Define neural network model
model = pof.neuralNetwork(load = "test1")

#Run model
input = pod.loadImage("example_image_one.png", (20,20), "grayscale")
output = model.run(input)
print(output)
```

## 2. examples/mnist
Training script for the full mnist dataset, 2 epochs -> less than 20 seconds on 12600k -> 96%+ accuracy on 10k-test dataset
```python
from pileoffeather import pof, pod

#Define neural network model
model = pof.neuralNetwork(layers = [[784,""],[128,"relu"],[10,"sigmoid"]], name = "mnist")

#Upload mnist dataset
X = pod.load(data_type = "gz", path = "train-images-idx3-ubyte.gz", start_index = 16, input_number = 784, divide = 255)
Y = pod.load(data_type = "gz", path = "train-labels-idx1-ubyte.gz", start_index = 8, one_hot = 10)

#Train the neural network using backpropagation
pof.train(model, X, Y, batch_size = 12, epoch_number = 2, rate = 1)
```

---

# Documentation
The library is structured in 2 files, pof.py (pile of feather) is used to create and train neural networks, pod.py (pile of data) is used import your own data to generate a training dataset.
## pof.py - neural network module
```
train(model, data_input, data_output, *batch_size, *epoch_number, *rate)

class neuralNetwork
            |
          __init__(*name, *layers, *load)
            |
          run(input)
            |
          load(name)
            |
          save()
            |
          initializeWeightsAndBiases()
            |
          computeBatch(batch_input, batch_output, batch_size, learning_rate)
            |
          pop(index)
            |
          insert(model, index)
```


```python
#Import module
from pileoffeather import pof

#Define neural network model, the available activation functions are "sigmoid","relu","leakyRelu"
model = pof.neuralNetwork(layers = [[400,""],[50,"relu"],[10,"relu"],[1,"sigmoid"]], name = "test1")

#Save the model
model.save()

#Load an exsisting model
model = pof.neuralNetwork(load = "test1")

#Use the neural network
output = model.run(input)

#Compute single mini_batch pass, model.computeBatch(batch_input, batch_output, batch_size, learning_rate)
model.computeBatch(batch_input, batch_output, 16, 0.3)

#Complete backpropagation over all the dataset, uses model.computeBatch in a loop
pof.train(model, input_matrix, output_matrix, batch_size = 16, epoch_number = 100, rate = 0.03)

#Pop a layer out of the neural network model (ex. remove layer at index 2)
model.pop(2)

#Insert a model layers over an other model, training is preserved (ex. at index 2)
model.insert(model1,2)
```

## pod.py - data load module

```python
#Import module
from pileoffeather import pod

#Load dataset of images from local folder
dataset = pod.load(data_type = "image", color = "grayscale", folder = "folder_name_containing_all_images", resize = (20,20))

#Load training input data of mnist, normalize input from 0 to 1 using divide = 255
dataset = pod.load(data_type = "gz", path = "train-images-idx3-ubyte.gz", start_index = 16, input_number = 784, divide = 255)

#Load training output data of mnist, use one_hot encoding to convert a decimal number to an array
#4 -> [0,0,0,0,1,0,0,0,0,0] 0 -> [1,0,0,0,0,0,0,0,0,0]
dataset = pod.load(data_type = "gz", path = "train-labels-idx1-ubyte.gz", start_index = 8, one_hot = 10)

#Load a single image to feed the neural network loadImage(name, resize, color)
input = pod.loadImage("example_image.png", (20,20), "grayscale")

#Convert neural network output to image and save, saveImage(neural_network_output, image_path, resize, color)
pod.saveImage(neural_network_output, "image_path_and_name", (20,20), "grayscale")
```

---

### TODO
- Improve pod (pileofdata) with better syntax and more data loading functions
- Add recurrent neural networks and common architectures like transformer or gan
- Implement a training method for those architectures
- Possibility to live graph loss, accuracy or similar stats during training
