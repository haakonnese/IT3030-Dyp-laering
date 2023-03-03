import numpy as np

from layers import BaseLayer


class Softmax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.activation_cache = None

    def forward_pass(self, C: np.ndarray):
        # C = np.clip(C, -500, 500)
        # np.seterr(all="raise")
        C_rel = C - np.max(C, axis=0)
        C_rel = np.exp(C_rel)
        sum_e = np.sum(C_rel, axis=0)
        a = C_rel / sum_e
        self.activation_cache = a
        return a

    def backward_pass(self, J: np.ndarray):
        """The derivative of the loss function with respect to the input of the softmax layer"""
        return J
