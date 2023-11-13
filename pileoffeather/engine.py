import numpy as np
import threading

class neuralNetworkModel:
    def __init__(self):
        self.name = "unknown"

    def _run(self, input):
        layer_history = np.empty(len(self.layers), dtype=object)
        layer_history[0] = input
        for layer in range(1,len(self.layers)):
            layer_history[layer] = np.add(np.dot(self.weights[layer], layer_history[layer-1]), self.biases[layer])
            match self.layers[layer][1]:
                case "sigmoid": layer_history[layer] = 1 / (1 + np.exp(-layer_history[layer]))
                case "relu": layer_history[layer] = np.maximum(0, layer_history[layer])
                case "leakyRelu": layer_history[layer] = np.maximum(0.1 * layer_history[layer], layer_history[layer])
        return layer_history

    def _backpropagate(self, run_history, output, calculate_loss):
        layer_number = len(self.layers)
        error = np.subtract(run_history[-1], output)
        for layer in range(layer_number - 1, 0, -1):
            match self.layers[layer][1]:
                case "sigmoid": delta = np.multiply(error, run_history[layer] * (1 - run_history[layer]))
                case "relu": delta = np.multiply(error, np.where(run_history[layer] > 0, 1, 0))
                case "leakyRelu": delta = np.multiply(error, np.where(run_history[layer] > 0, 1, 0.1))
            with threading.Lock():
                self.weights_mod[layer] += np.outer(delta, run_history[layer - 1])
                self.biases_mod[layer] += delta
            if layer>1: error = np.dot(self.weights[layer].T, delta)
        if calculate_loss:
            self.loss = abs(np.mean(run_history[-1] - output))

    def _update_weights(self, batch_size, learning_rate):
        factor = 1/batch_size*learning_rate
        self.weights -= self.weights_mod * factor
        self.biases -= self.biases_mod * factor
        self.weights_mod, self.biases_mod, self.loss = np.zeros_like(self.weights), np.zeros_like(self.biases), 0

    def _init_weights(self):
        self.weights, self.biases = np.zeros(len(self.layers), dtype=object), np.zeros(len(self.layers), dtype=object)
        for i in range(1, len(self.layers)):
            match self.layers[i][1]:
                case "sigmoid": self.weights[i] = np.random.uniform(low=-0.1, high=0.1, size=(self.layers[i][0], self.layers[i-1][0])).astype(np.float32)
                case "relu": self.weights[i] = np.random.normal(loc=0, scale=np.sqrt(2 / self.layers[i-1][0]), size=(self.layers[i][0], self.layers[i-1][0])).astype(np.float32)
                case "leakyRelu": self.weights[i] = np.random.normal(loc=0, scale=np.sqrt(2 / self.layers[i-1][0]), size=(self.layers[i][0], self.layers[i-1][0]))*np.where(self.weights[i] > 0, 1, 0.1).astype(np.float32)
            self.biases[i] = np.zeros(self.layers[i][0], dtype=np.float32)
        self.weights_mod, self.biases_mod, self.loss = np.zeros_like(self.weights), np.zeros_like(self.biases), 0

    def run(self, input):
        return self._run(input)[-1]

    def save(self):
        np.savez(self.name + ".npz", matrix1=self.weights, matrix2=self.biases, matrix3=self.layers)

    def load(self):
        data = np.load(self.name + ".npz", allow_pickle=True)
        self.weights, self.biases, self.layers = data['matrix1'], data['matrix2'], [[int(item) if item.isdigit() else item for item in row] for row in data['matrix3']]
        self.weights_mod, self.biases_mod, self.loss = np.zeros_like(self.weights), np.zeros_like(self.biases), 0

    def pop(self, index=-1):
        self.layers.pop(index)
        self.weights = np.delete(self.weights, index, axis=0)
        self.biases = np.delete(self.biases, index, axis=0)

    def insert(self, model, index=-1):
        self.layers[index:index] = model.layers
        self.weights = np.insert(self.weights, index, model.weights, axis=0)
        self.biases = np.insert(self.biases, index, model.biases, axis=0)
