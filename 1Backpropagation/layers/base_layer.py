from abc import abstractmethod

import numpy as np


class BaseLayer:
    @abstractmethod
    def forward_pass(self, C: np.ndarray):
        pass

    @abstractmethod
    def backward_pass(self, J: float):
        pass
