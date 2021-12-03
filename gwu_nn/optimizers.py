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
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.updated_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    
    # Called once before the optimize function
    def before_optimize(self):
        if self.decay:
            self.updated_learning_rate = self.learning_rate * (1.0 / (1.0 + (self.decay * self.iterations)))

    # Optimize (Update Layer Weights/Bias)
    def optimize(self, layer):

        if self.momentum:

            # Create momentums if not already present
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            weight_updates = self.momentum * layer.weight_momentums - self.updated_learning_rate * layer.weights_error
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = self.momentum * layer.bias_momentums - self.updated_learning_rate * layer.output_error
            layer.bias_momentums = bias_updates

        else:
                
            layer.weights += -self.updated_learning_rate * layer.weights_error
            if layer.add_bias:
                layer.bias -= self.updated_learning_rate * layer.output_error

    # Called once after optimize function
    def after_optimize(self):
        self.iterations += 1


# Adagrad optimizer
class Adagrad(Optimizer):

    # Initialize optimizer
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-5):
        self.learning_rate = learning_rate
        self.updated_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Called once before the optimize function
    def before_optimize(self):
        if self.decay:
            self.updated_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Optimize (Update Layer Weights/Bias)
    def optimize(self, layer):

        # Create caches if not already present
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.bias)

        # Update the layer caches
        layer.weight_cache += layer.weights_error**2
        layer.bias_cache += layer.output_error**2

        # Parameter updates (Modified SGD)
        layer.weights += -self.updated_learning_rate * layer.weights_error / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.bias += -self.updated_learning_rate * layer.output_error / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Called once after optimize function
    def after_optimize(self):
        self.iterations += 1



