from littlenn.layers import Conv2D, Flatten
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

