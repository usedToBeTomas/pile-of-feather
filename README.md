<div align="center">
<h1>POF - pile of feather</h1>
<img src="https://github.com/usedToBeTomas/pile-of-feather/blob/main/images/pof.png" width="400" height="400" />

Lightweight and easy to use neural network library for small projects, create a neural network in minutes. A fun project.

<h3>

[Homepage](https://github.com/usedToBeTomas/pile-of-feather) | [Documentation](https://github.com/usedToBeTomas/pile-of-feather#documentation-book) | [Examples](https://github.com/usedToBeTomas/pile-of-feather#examples-rocket)

</h3>

</div>

---

Install module
```cmd
pip install pileoffeather
```
# Examples :rocket:
Handwritten digit image classifier, dataset is made out of 500 images of ones and 500 images of zeros taken from the mnist dataset. The first code snippet is defining the neural network model, uploading the dataset and than training the model
```python
from pileoffeather import pof, pod
import numpy as np

#Define neural network model
model = pof.neuralNetwork(layers = [[400,"input"],[30,"relu"],[10,"relu"],[1,"sigmoid"]], name = "test1")

#Load the images from their respective folders, 500 ones images and 500 zeros images
ones = pod.load(data_type = "image", folder = "ones", resize = (20,20))
zeros = pod.load(data_type = "image", folder = "zeros", resize = (20,20))
input = np.vstack((ones, zeros)) #Merge the ones and zeros

#Generate expected output, first 500 images should output 1, the other 500 0
output = np.concatenate((np.ones(500), np.zeros(500))) #Merge the ones and zeros

#Train the neural network using backpropagation, it automatically saves the weights and biases at the end of the 100 epochs
model.train(input, output, batch_size = 16, epoch_number = 100, rate = 0.6)
```
The second code snippet is testing the neural network on some examples
```python
from pileoffeather import pof, pod
import numpy as np

#Define neural network model
model = pof.neuralNetwork(load = "test1")

#Run model
output = model.run(pod.loadImage("example_image_one.png", (20,20)))

#Print result
print(round(output[0],3))

```

---

# Documentation :book:
The library is structured in 2 files, pof.py (pile of feather) is the library for creating and training neural networks and pod.py (pile of data) is the library for importing your own data to create a training dataset for the neural network.
## pof.py - neural network module
Install module
```cmd
pip install pileoffeather
```
Import module
```python
from pileoffeather import pof
```
Define neural network model, the available activation functions are "sigmoid","relu","leakyRelu"
```python
model = pof.neuralNetwork(layers = [[400,""],[50,"relu"],[10,"relu"],[1,"sigmoid"]], name = "test1")
```
Load an exsisting model
```python
model = pof.neuralNetwork(load = "test1")
```
Save the model
```python
model.save()
```
Train using backpropagation
```python
model.train(input_matrix, output_matrix, batch_size = 16, epoch_number = 100, rate = 0.03)
```
Use the neural network
```python
output = model.run(input)
```

## pod.py - data load module
Import module
```python
from pileoffeather import pod
```
Load images from a folder
```python
ones = pod.load(data_type = "image", folder = "folder_name_containing_all_images", resize = (20,20))
```

