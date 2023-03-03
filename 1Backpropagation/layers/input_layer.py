import numpy as np

from layers import LayerWithSize


class Input(LayerWithSize):
    def __init__(self, size: int):
        super().__init__(size)

    def forward_pass(self, C: np.ndarray):
        if C.shape[0] != self.size:
            raise ValueError("The input is not of the same dimension as the input layer. The input is of size "
                             f"{C.shape[0]}, while the input layer has size {self.size}")
        return C

    def backward_pass(self, loss: np.ndarray):
        pass
