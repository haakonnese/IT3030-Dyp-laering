import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import layers
from general import LossFunction, WeightRegularizationType


class Sequential:
    def __init__(self):
        self.fitted = False
        self.weight_regularization_rate = None
        self.loss = None
        self.regularization = None
        self.learning_rate = None
        self.layers: list[layers.BaseLayer] = []
        self.seed: int | None = None
        self.iterations: int = 1000
        self.batch_size: int = 1

    def add(self, layer: layers.BaseLayer):
        if len(self.layers) == 0 and type(layer) is not layers.Input:
            raise ValueError("The first layer must be an input layer")
        elif len(self.layers) > 0 and type(layer) is layers.Input:
            raise ValueError("Only the first layer can be an input layer")
        self.layers.append(layer)

    def compile(self,
                learning_rate: float = 0.1,
                loss: LossFunction = LossFunction.SQUARED_ERROR,
                regularization: WeightRegularizationType = WeightRegularizationType.L1,
                weight_regularization_rate: float = 0.001,
                random_state: int | None = None,
                iterations: int | None = None,
                batch_size: int | None = None):
        self.learning_rate = learning_rate
        self.seed = random_state
        if random_state is not None:
            np.random.seed(random_state)
        if iterations is not None:
            self.iterations = iterations
        if batch_size is not None:
            self.batch_size = batch_size
        if isinstance(self.layers[-1], layers.Softmax):
            second_last_layer = self.layers[-2]
            if isinstance(second_last_layer, layers.LayerWithSize):
                if second_last_layer.size == 1:
                    raise ValueError("The output layer must have at least two neurons when using softmax")
            if loss == LossFunction.SQUARED_ERROR:
                raise ValueError("`squared_error` is not supported with categorical data.")
        self.loss = loss
        if regularization == WeightRegularizationType.L1:
            self.regularization = lambda x: np.sign(x)
        elif regularization == WeightRegularizationType.L2:
            self.regularization = lambda x: x
        self.weight_regularization_rate = weight_regularization_rate

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, verbose: int = 0):
        """Fit the model to the data
        :param X: The input data
        :param y: The target data
        :param X_val: The validation input data
        :param y_val: The validation target data
        :param verbose: 0: no output, 1: output input, output, target and loss, 2: same as 1 but print the whole and not
         truncated arrays"""
        X = X.T
        X_val = X_val.T
        self._initialize_weights()
        loss = []
        val_loss = []
        for i in range(self.iterations):
            for j in range(0, X.shape[1], self.batch_size):
                predictions = self._forward_pass(X[:, j:j+self.batch_size])
                loss.append(self._loss(y[j:j+self.batch_size], predictions))
                J = self._derivative_loss(y[j:j+self.batch_size], predictions)
                self._backward_pass(J)
                val_predictions = self._forward_pass(X_val)
                val_loss.append(self._loss(y_val, val_predictions))
        if verbose:
            if verbose == 2:
                import sys
                np.set_printoptions(threshold=sys.maxsize)
            print("Network input:", X)
            predictions = self._forward_pass(X)
            predictions_val = self._forward_pass(X_val)
            print("Network output:", predictions.T)
            print("Network loss training:", self._loss(y, predictions))
            print("Network loss validation:", self._loss(y_val, predictions_val))
            print("Y_true:", y)
        self.fitted = True
        sns.lineplot(data=loss, label="Training loss")
        sns.lineplot(data=val_loss, label="Validation loss")
        plt.show()
        return self

    def predict(self, X: np.ndarray):
        if not self.fitted:
            raise AttributeError("The model is not fitted yet")
        return self._forward_pass(X)

    def _forward_pass(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward_pass(X)
        return X

    def _backward_pass(self, J: float):
        for layer in reversed(self.layers):
            J = layer.backward_pass(J)

    def _initialize_weights(self):
        # initialize weights, possibly with glorot which requires the size of the previous layer
        last_layer_num_nodes = None
        for layer in self.layers:
            if isinstance(layer, layers.Input):
                last_layer_num_nodes = layer.size
            if isinstance(layer, layers.Dense):
                layer.set_learning_rate(self.learning_rate)
                layer.initialize_weights(last_layer_num_nodes, self.seed)
                layer.set_regularization(self.regularization, self.weight_regularization_rate)
                last_layer_num_nodes = layer.size

    def _loss(self, y: np.ndarray, predictions: np.ndarray):
        if self.loss == LossFunction.CROSS_ENTROPY:
            # ensure that the predictions has no 0's to avoid zero division
            predictions = np.clip(predictions, 1e-7, None)
            loss = -np.mean(np.sum(y * np.log(predictions.T), axis=1))
            return loss
        elif self.loss == LossFunction.SQUARED_ERROR:
            return np.mean(np.square(y - predictions))

    def _derivative_loss(self, y: np.ndarray, predictions: np.ndarray):
        if self.loss == LossFunction.CROSS_ENTROPY:
            """The derivative of the categorical cross entropy loss function with respect to the output of the softmax layer"""
            if isinstance(self.layers[-1], layers.Softmax):
                return (predictions - y.T) / y.shape[1]
            else:
                raise ValueError("The derivative of the categorical cross entropy loss "
                                 "function is only supported with softmax")

        elif self.loss == LossFunction.SQUARED_ERROR:
            return 2 / predictions.shape[1] * (predictions - y)
