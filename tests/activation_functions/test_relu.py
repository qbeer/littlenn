from littlenn.activation_functions.relu import ReLU
import numpy as np

test_input = np.random.randn(10, 10)

relu = ReLU()

relu_output = relu(test_input)

assert relu_output.all() >= 0., "All outputs of ReLU must be larger or equal to zero."

relu_deriv = relu.derivative(relu_output)

assert np.abs(relu_deriv.all() - 1.).all() == (0 or 1), "All derivates must be constant 1 or 0 for ReLU"
