import numpy as np

from layers import LayerWithSize
from general import ActivationFunctions, GLOROT, relu, identity, \
    sigmoid, der_relu, der_sigmoid, der_tanh, der_identity, WeightRegularizationType


class Dense(LayerWithSize):
    def __init__(self,
                 size: int,
                 activation: ActivationFunctions = ActivationFunctions.RELU,
                 initial_weight_range: str | tuple[float, float] = GLOROT,
                 learning_rate: float = None,
                 bias_range: tuple[float, float] | None = None
                 ):
        """
        A fully connected layer

        :param size: Number of nodes in the layer
        :param activation: the activation function of the layer. Supports `identity`, `logistic`, `tanh` and `relu`.
        Default is `relu`
        :param initial_weight_range: the initial range of the weights. Supports the `glorot` algorithm form initializing
        the weights, and a tuple of two floats, e.g. (-0.1, 0.1). Default is `glorot`
        :param learning_rate: of the layer. If none is specified, the global learning rate will be used. Default, use
        global learning rate of the network
        :param bias_range: the range of the biases of the neurons in the layer. E.g. (0, 1)
        """
        super().__init__(size)
        self.activation_cache = None
        if activation.upper() == ActivationFunctions.RELU:
            self.activation = relu
            self.derivative_activation = der_relu
        elif activation.upper() == ActivationFunctions.LOGISTIC:
            self.activation = sigmoid
            self.derivative_activation = der_sigmoid
        elif activation.upper() == ActivationFunctions.IDENTITY:
            self.activation = identity
            self.derivative_activation = der_identity
        elif activation.upper() == ActivationFunctions.TANH:
            self.activation = np.tanh
            self.derivative_activation = der_tanh
        else:
            raise ValueError("activation argument supports values `identity`, `logistic`, `tanh` and `relu`")
        if type(initial_weight_range) == str and initial_weight_range != GLOROT:
            raise ValueError("Supports the `glorot` algorithm form initializing the weights, "
                             f"and a tuple of two floats, e.g. (-0.1, 0.1), but not {initial_weight_range}")
        if bias_range is not None:
            if len(bias_range) != 2:
                raise ValueError(f"`bias_range` has length {len(bias_range)} but requires length 2")
            elif bias_range[0] > bias_range[1]:
                raise ValueError(f"`bias_range` needs to be in ascending order")
        else:
            bias_range = (float("-inf"), float("inf"))
        self.initial_weight_range = initial_weight_range
        self.learning_rate = learning_rate
        self.bias_range = bias_range
        self.bias = np.zeros((size, 1))
        self.inputs: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.regularization = lambda x: 1
        self.regularization_rate: float = 0

    def forward_pass(self, C: np.ndarray):
        self.inputs = C
        if isinstance(self.W, np.ndarray):
            # print(f"{self.W.shape = }")
            # print(f"{C.shape = }")
            # print(f"{self.bias.shape = }")
            # print(f"{np.repeat(self.bias, 1, axis=1).shape = }")
            # print(f"{np.dot(self.W.T, C).shape = }")
            try:
                size = C.shape[1]
            except IndexError:
                size = 1
            self.activation_cache = \
                self.activation(np.dot(self.W, C) + np.repeat(self.bias.reshape((-1,1)), size, axis=1))
            return self.activation_cache
        raise AttributeError("Weights are None, causing error")

    def backward_pass(self, J: np.ndarray):
        # Compute the gradient of the weights
        dDelta = J * self.derivative_activation(self.activation_cache)
        dW = np.dot(dDelta, self.inputs.T)

        # Compute the gradient of the biases
        dB = np.mean(dDelta, axis=1, keepdims=True)

        # Compute the error to be passed back to the previous layer
        # Update the weights and biases
        self._update_weights_and_bias(dW, dB)
        return np.dot(self.W.T, dDelta)

    def initialize_weights(self, input_nodes: int, random_state: int | None = None):
        if random_state is not None:
            np.random.seed(random_state)
        # print(random_state)
        if input_nodes is None:
            raise AttributeError("Input nodes is None")
        if self.initial_weight_range == GLOROT:
            sd = np.sqrt(2 / (input_nodes + self.size))
            self.W = np.random.normal(0.0, sd, size=(self.size, input_nodes))
        else:
            self.W = np.random.uniform(self.initial_weight_range[0],
                                       self.initial_weight_range[1],
                                       size=(self.size, input_nodes),
                                       )
        # print(self.W)

    def _update_weights_and_bias(self, dW, dB):
        self.W -= self.learning_rate * (dW + self.regularization(self.W) * self.regularization_rate)
        # ensure bias is in configured range
        self.bias = np.clip(self.bias - self.learning_rate *
                            (dB + self.regularization(self.bias) * self.regularization_rate),
                            self.bias_range[0], self.bias_range[1])

    def set_learning_rate(self, learning_rate: float):
        if self.learning_rate is None:
            self.learning_rate = learning_rate

    def set_regularization(self, regularization, regularization_rate: float):
        self.regularization = regularization
        self.regularization_rate = regularization_rate
