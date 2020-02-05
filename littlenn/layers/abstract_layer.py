import abc
from littlenn.activation_functions import ReLU, Sigmoid, Linear

class Layer(abc.ABC):
    def __init__(self):
        self.activations = {"relu" : ReLU(), "sigmoid" : Sigmoid(),
                            "linear" : Linear(), "None" : Linear()}
    
    @abc.abstractmethod  
    def _create_weights(self):
        pass

    @abc.abstractmethod
    def _get_weights(self):
        pass

    @abc.abstractmethod
    def _get_trainable_params(self):
        pass

    @abc.abstractmethod
    def __call__(self, x):
        pass

    @abc.abstractmethod
    def grads(self, x):
        pass

    @abc.abstractmethod
    def _apply_grads(self, x):
        pass