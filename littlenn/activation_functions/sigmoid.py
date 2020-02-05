from abstract_activation import ActivationFunction

class Sigmoid(ActivationFunction):
    def __init__(self):
        self.sigmoid = lambda x : 1. / (1. + np.exp(-x))
    def __call__(self, x):
        return self.sigmoid(x)
    def backprop(self, x):
        return self.sigmoid(x) * self.sigmoid(-x)