import abc
from littlenn.optimizers.sgd_factory import SGDFactory

class Model(abc.ABC):
    def __init__(self):
        self.optimizers = {"sgd" : SGDFactory()}