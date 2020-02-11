from littlenn.layers import Dropout
import numpy as np

dropout = Dropout(keep_prob=0.8)
dropout._create_weights(1000)

x = np.random.randn(1000, 10)

y = dropout(x)

print(np.where(dropout.W)[0].size / dropout.W.size)
print(np.sum(dropout.W == 0).size / dropout.W.size)

print(y.shape)