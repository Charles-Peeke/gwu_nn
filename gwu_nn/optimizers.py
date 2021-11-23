import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    @abstractmethod
    def update_params(self, layer):
        pass
