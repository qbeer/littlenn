from littlenn.optimizers.sgd import SGD

class SGDFactory:
    def create_instance(self, learning_rate, exponential_weight):
        return SGD(learning_rate, exponential_weight)

