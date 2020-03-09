from littlenn.layers import Flatten
import numpy as np

a = np.random.randn(256, 14, 14, 16)

flat = Flatten()

y = flat(a, True)

print(y.shape)