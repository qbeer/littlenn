from abstract_activation import ActivationFunction

class Linear(ActivationFunction):
    def __call__(self, x):
        return x
    def backprop(self, x):
        return x