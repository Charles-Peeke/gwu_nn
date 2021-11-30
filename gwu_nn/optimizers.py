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
    def __init__(self, learning_rate=1.0, decay=0.0):
        self.learning_rate = learning_rate
        self.updated_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
    
    # Called once before the optimize function
    def before_optimize(self):
        if self.decay:
            self.updated_learning_rate = self.learning_rate * (1.0 / (1.0 + (self.decay * self.iterations)))

    # Optimize (Update Layer Weights/Bias)
    def optimize(self, layer):
        layer.weights += -self.updated_learning_rate * self.weights_error
        if layer.add_bias:
            layer.bias -= self.updated_learning_rate * self.output_error

    # Called once after optimize function
    def after_optimize(self):
        self.iterations += 1

