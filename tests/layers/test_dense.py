from littlenn.layers import Dense
import numpy as np

DIM_IN = 100
DIM_OUT = 10
SIZE = 100 * 10 + 10

dense = Dense(dim_out=DIM_OUT)
dense._create_weights(DIM_IN)

weights = dense._get_weights()

assert weights.size == SIZE, 'Weight should be equal to init params.'

x = np.random.randn(100, 10) # a mini-batch of inputs

y = dense(x, True)

assert y.shape == (10, 10), 'Shape must be equal to expected output shape'