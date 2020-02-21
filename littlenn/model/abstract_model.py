import abc
from littlenn.optimizers.sgd_factory import SGDFactory
from littlenn.optimizers.rmsprop_factory import RMSPropFactory
from littlenn.optimizers.adam_factory import AdamFactory

class Model(abc.ABC):
    def __init__(self):
        self.optimizers = {"sgd" : SGDFactory(), "rmsprop" : RMSPropFactory(),
                           "adam" : AdamFactory()}