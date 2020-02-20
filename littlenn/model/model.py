import numpy as np
from littlenn.model.abstract_model import Model

class Sequential(Model):
    def __init__(self, input_size, optimizer_params):
        super(Sequential, self).__init__()
        self.layers = []
        self.input_size = input_size
        self.__current_input_size = input_size
        self.learning_rate = optimizer_params['lr']
        self.exponential_weight = optimizer_params['ew']
        self.optimizer_factory = self.optimizers[optimizer_params['name']]

    def add(self, layer):
        layer._create_weights(self.__current_input_size)
        layer._init_optimizers(self.optimizer_factory, self.learning_rate, self.exponential_weight)

        self.layers.append(layer)
        self.__current_input_size = layer.dim_out

    def __call__(self, x, training=True):
        for layer in self.layers:
            x = layer(x, training)         
        return x

    def _backprop(self, grads):
        for layer in self.layers[::-1]:
            grads = layer.grads(grads)
            layer._apply_grads(grads)

    def summary(self):
        param_count = 0
        print('MODEL:')
        for layer in self.layers:
            param_count +=  layer._get_trainable_params()
            print(layer, '\ttrainable params : %d' % (layer._get_trainable_params()))
        print('Total trainable parameters : %d' % param_count)
    
    def fit(self):
        pass