from enum import Enum

import numpy as np


class LossFunction(str, Enum):
    SQUARED_ERROR = "SQUARED_ERROR"
    CROSS_ENTROPY = "CROSS_ENTROPY"


class WeightRegularizationType(str, Enum):
    L1 = "L1"
    L2 = "L2"


class ActivationFunctions(str, Enum):
    IDENTITY = "IDENTITY"
    LOGISTIC = "LOGISTIC"
    TANH = "TANH"
    RELU = "RELU"


def relu(x: np.ndarray):
    x[x < 0] = 0
    return x


def der_relu(activation_cache: np.ndarray):
    derivative = np.where(activation_cache != 0, 1, 0)
    return derivative


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def der_sigmoid(activation_cache: np.ndarray):
    return activation_cache * (1 - activation_cache)


def identity(x: np.ndarray):
    return x


def der_identity(activation_cache: np.ndarray):
    return 1


def der_tanh(activation_cache: np.ndarray):
    return 1 - activation_cache ** 2


DEFAULT_SIZE = 20
GLOROT = "glorot"