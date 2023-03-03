from abc import ABC

from layers import BaseLayer


class LayerWithSize(BaseLayer, ABC):
    def __init__(self, size):
        self.size = size
