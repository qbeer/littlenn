from littlenn.layers import Conv2D
import numpy as np

DIM_IN = (256, 14, 14)
C_OUT = 32

conv2d = Conv2D(channels_out=C_OUT, kernel_size=(3, 3),
                padding=(0, 0), strides=(1, 1),
                activation='relu')
conv2d._create_weights(DIM_IN)

x = np.random.randn(256, 14, 14, 10)
y = conv2d(x, True)

assert conv2d._get_weights().size == 3 * 3 * 256 * 32 + 32, 'Number of parameters in a convolutional layer must correspond to this'

dy = np.random.randn(256, 14, 14, 10)

_, dK, db = conv2d.grads((dy, None, None))

assert dK.shape == conv2d.kernels.shape and db.shape == conv2d.biases.shape, 'Derivates of paremeters must be the same shape as the params'