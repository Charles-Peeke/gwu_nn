import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    @abstractmethod
    def before_optimize(self):
        pass

    @abstractmethod
    def optimize(self, layer):
        pass

    @abstractmethod
    def after_optimize(self):
        pass

# SGD optimizer
class SGD(Optimizer):

    # Initialize optimizer - 
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    # Optimize (Update Layer Weights/Bias)
    def optimize(self, layer):
        layer.weights += -self.learning_rate * self.weights_error
        if layer.add_bias:
            layer.bias -= self.learning_rate * self.output_error

