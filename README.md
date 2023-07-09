<div align="center">
<h1>POF - pile of feather</h1>
<img src="https://github.com/usedToBeTomas/pile-of-feather/blob/main/images/pof.png" width="400" height="400" />

Lightweight and easy to use neural network library for small projects, create a neural network in minutes. A fun project.

<h3>

[Homepage](https://github.com/usedToBeTomas/pile-of-feather) | [Documentation](https://github.com/usedToBeTomas/pile-of-feather#documentation) | [Examples](https://github.com/usedToBeTomas/pile-of-feather#examples)

</h3>

</div>

# Documentation
Import module
```python
import pof
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

# Examples
Working in progress...

