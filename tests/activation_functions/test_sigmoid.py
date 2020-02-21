from littlenn.activation_functions.sigmoid import Sigmoid
import numpy as np

test_input = np.random.randn(10, 10)

sigmoid = Sigmoid()

sigmoid_out = sigmoid(test_input)

negative_indices = test_input < 0.
positive_indices = test_input >= 0.

assert (sigmoid_out[positive_indices] >= 0.5).any() and (sigmoid_out[negative_indices] < 0.5).any(), "Sigmoid should map negative values below 0.5 and positive values above 0.5"

sigmoid_deriv = sigmoid.derivative(sigmoid_out)

assert (sigmoid_deriv == sigmoid(sigmoid_out) * sigmoid(-sigmoid_out)).all(), "Sigmoid derivative is itself times itself by the negative value"