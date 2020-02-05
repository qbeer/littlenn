from abstract_activation import ActivationFunction

class ReLU(ActivationFunction):
    def __call__(self, x):
        return np.maximum(x, 0)
    def backprop(self, x):
        return np.maximum(x / np.abs(x + 1e-12), 0)