import numpy as np, nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = np.random.randn(n_inputs, n_neurons) / 100
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs.dot(self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0: self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0: self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        self.dinputs = dvalues.dot(self.weights.T)

class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs, training):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs


class Activation_ReLU:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs

class Activation_Softmax:
    def forward(self, inputs, training):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, 1, keepdims=True))
        self.output = exp_values / exp_values.sum(1, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            self.dinputs[index] = (np.diagflat(single_output) - single_output.dot(single_output.T)).dot(single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, 1)

class Activation_Sigmoid:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return outputs > 0.5

class Activation_Linear:
    def forward(self, inputs, training):
        self.inputs = self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs


class Optimizer_SGD:
    def __init__(self, learning_rate=1, decay=0, momentum=0):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay: self.learning_rate /= 1 + self.decay * self.iterations

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.learning_rate * layer.dweights
            bias_updates = -self.learning_rate * layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adagrad:
    def __init__(self, learning_rate=1, decay=0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay: self.learning_rate /= 1 + self.decay * self.iterations

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        layer.weights -= self.learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases -= self.learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimizer_RMSprop:
    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay: self.learning_rate /= 1 + self.decay * self.iterations

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2
        layer.weights -= self.learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases -= self.learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay: self.learning_rate /= 1 + self.decay * self.iterations

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        layer.weights -= self.current_learning_rate * layer.weight_momentums / ((1 - self.beta_1**(self.iterations + 1)) * (np.sqrt(layer.weight_cache / (1 - self.beta_2**(self.iterations + 1))) + self.epsilon))
        layer.biases -= self.current_learning_rate * layer.bias_momentums / ((1 - self.beta_1**(self.iterations + 1)) * (np.sqrt(layer.bias_cache / (1 - self.beta_2**(self.iterations + 1))) + self.epsilon))

    def post_update_params(self):
        self.iterations += 1


class Loss:
    def forward(self, output, y): pass

    def backward(self, dvalues, y_true): pass

    def regularization_loss(self):
        regularization_loss = 0
        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0: regularization_loss += layer.weight_regularizer_l1 * np.abs(layer.weights).sum()
            if layer.weight_regularizer_l2 > 0: regularization_loss += layer.weight_regularizer_l2 * (layer.weights**2).sum()
            if layer.bias_regularizer_l1 > 0: regularization_loss += layer.bias_regularizer_l1 * np.abs(layer.biases).sum()
            if layer.bias_regularizer_l2 > 0:regularization_loss += layer.bias_regularizer_l2 * (layer.biases**2).sum()
        return regularization_loss

    def remember_trainable_layers(self, trainable_layers: list[Layer_Dense]):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False):
        data_loss = self.forward(output, y).mean()
        return data_loss if not include_regularization else data_loss, self.regularization_loss()

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1: correct_confidences = y_pred_clipped[range(len(y_pred)), y_true]
        elif len(y_true.shape) == 2: correct_confidences = (y_pred_clipped * y_true).sum(1)
        return -np.log(correct_confidences)

    def backward(self, dvalues, y_true):
        if len(y_true.shape) == 1: y_true = np.eye(len(dvalues[0]))[y_true]
        self.dinputs = -y_true / (dvalues * len(dvalues))

class Activation_Softmax_Loss_CategoricalCrossentropy:
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2: y_true = np.argmax(y_true, 1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples

class Loss_BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)).mean(-1)

    def backward(self, dvalues, y_true):
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / (len(dvalues[0]) * len(dvalues))

class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        return ((y_true - y_pred)**2).mean(-1)

    def backward(self, dvalues, y_true):
        self.dinputs = -2 * (y_true - dvalues) / (len(dvalues[0]) * len(dvalues))

class Loss_MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        return np.abs(y_true - y_pred).mean(-1)

    def backward(self, dvalues, y_true):
        self.dinputs = np.sign(y_true - dvalues) / (len(dvalues[0]) * len(dvalues))


class Accuracy:
    def init(self, y, reinit): pass

    def compare(self, predictions, y): pass

    def calculate(self, predictions, y):
        return self.compare(predictions, y).mean()

class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        self.binary = binary

    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2: y = np.argmax(y, 1)
        return predictions == y

class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision or reinit: self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision


class Model:
    def __init__(self):
        self.layers: list[Layer_Dense | Layer_Dropout | Layer_Input] = []
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        self.input_layer = Layer_Input()
        self.trainable_layers = []
        for i, layer in enumerate(self.layers):
            layer.prev = self.input_layer if i == 0 else self.layers[i - 1]
            layer.next = self.layers[i + 1] if i == 0 or i < len(self.layers) - 1 else self.loss
            self.output_layer_activation = layer
            if hasattr(layer, 'weights'): self.trainable_layers.append(layer)
        self.loss.remember_trainable_layers(self.trainable_layers)
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy): self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):
        self.accuracy.init(y)
        for epoch in range(1, epochs + 1):
            output = self.forward(X, True)
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            self.backward(output, y)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers: self.optimizer.update_params(layer)
            self.optimizer.post_update_params()
            if not epoch % print_every: print(f'epoch: {epoch}, acc: {self.accuracy.calculate(self.output_layer_activation.predictions(output), y):.3f}, loss: {data_loss + regularization_loss:.3f} (data_loss: {data_loss:.3f}, reg_loss: {regularization_loss:.3f}), lr: {self.optimizer.learning_rate}')
        if validation_data:
            output = self.forward(validation_data[0], False)
            print(f'validation, acc: {self.accuracy.calculate(self.output_layer_activation.predictions(output), validation_data[1]):.3f}, loss: {self.loss.calculate(output, validation_data[1]):.3f}')

    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers: layer.forward(layer.prev.output, training)
        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]): layer.backward(layer.next.dinputs)
            return
        self.loss.backward(output, y)
        for layer in reversed(self.layers): layer.backward(layer.next.dinputs)

X, y = spiral_data(1000, 3)
model = Model()
model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())
model.set(loss=Loss_CategoricalCrossentropy(), optimizer=Optimizer_Adam(0.05, 5e-5), accuracy=Accuracy_Categorical())
model.finalize()
model.train(X, y, epochs=10000, print_every=100, validation_data=spiral_data(100, 3))
