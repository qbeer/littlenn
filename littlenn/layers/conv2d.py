import numpy as np
from littlenn.layers.abstract_layer import Layer

class Conv2D(Layer):
    def __init__(self, channels_out, kernel_size,
                 padding, strides, activation=None):
        super(Conv2D, self).__init__()
        self.channels_out = channels_out
        self.act_fn = self.activations[activation]
        self.padding = padding
        self.strides = strides
        self.kernel_size = kernel_size

    def _create_weights(self, dim_in):
        self.channels_in, self.input_height, self.input_width = dim_in
        # Xavier-initialization
        self.kernels = np.random.randn(
            self.channels_in, self.kernel_size[0], self.kernel_size[1], self.channels_out) * np.sqrt(
                2 / (self.channels_in * self.kernel_size[0] * self.kernel_size[1] * self.channels_out))
        self.biases = np.random.randn(self.channels_out, 1) * np.sqrt(1 / (self.channels_out))

    def _init_optimizers(self, optimizer_factory, params):
        self.kernels_opt = optimizer_factory.create_instance(params)
        self.biases_opt = optimizer_factory.create_instance(params)

    def _get_weights(self):
        weights = np.concatenate((self.kernels.flatten(), self.biases.flatten()), axis=None)
        return weights.reshape(-1, 1)

    def _get_trainable_params(self):
        return self.kernels.size + self.biases.size

    def __correlation(self, x, kernel):
        _, h, w = kernel.shape
        C, H, W, bs = x.shape

        assert kernel.shape[0] == x.shape[0], 'Channels must be equal for kernel and input tensor'

        # H, and W already padded (!)
        H_out = np.floor(( H - h + self.strides[0] ) / self.strides[0]).astype(int)
        W_out = np.floor(( W - w + self.strides[1] ) / self.strides[1]).astype(int)
        
        Y = np.zeros((H_out, W_out, bs))
        
        for i in range(H, self.strides[0]):
            for j in range(W, self.strides[1]):
                Y[:, i, j, :] = np.sum(X[:, i:i+h, j:j+w, :] * K)
        
        return Y

    def __pad(self, x):
        return np.pad(
            x, ((0, 0), (self.padding[0], self.padding[0]),
                (self.padding[1], self.padding[1]), (0, 0)),
            mode='constant', constant_values=0)

    def __multichannel_correlation(self, x):
        return np.stack(
            [self.__correlation(
                self.__pad(x), k) for k in np.einsum('ijkl->lijk', self.kernels)])

    def __call__(self, x, training):
        output = self.__multichannel_correlation(x)
        return self.act_fn(output)

    def grads(self, grads):
        dprev, *_ = grads
        act_deriv = self.act_fn.derivative(self.act)
        dW = np.matmul(dprev * act_deriv, self.z_prev.T)
        db = np.mean(dprev * act_deriv, keepdims=True)
        dprev_new =  np.matmul(self.W.T, dprev * act_deriv)
        return dprev_new, dW, db

    def _apply_grads(self, grads):
        *_, dW, db = grads
        self.W = self.W_opt(self.W, dW)
        self.b = self.b_opt(self.b, db)

    def __str__(self):
        return "DenseBlock : (%s, %s)" % (self.W.shape[1], self.W.shape[0])