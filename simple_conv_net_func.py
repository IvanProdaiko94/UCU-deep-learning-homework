from __future__ import print_function
import torch


def diff_mse(x, y):
    x_vec = x.view(1, -1).squeeze()
    y_vec = y.view(1, -1).squeeze()
    return torch.mean(torch.pow((x_vec - y_vec), 2)).item()


def conv2d_scalar(x_in, conv_weight, conv_bias, device):
    #
    # Add your code here
    #
    pass


def conv2d_vector(x_in, conv_weight, conv_bias, device):
    #
    # Add your code here
    #
    pass


def im2col(X, kernel_size, device):
    #
    # Add your code here
    #
    pass


def conv_weight2rows(conv_weight):
    #
    # Add your code here
    #
    pass


def pool2d_scalar(a, device):
    #
    # Add your code here
    #
    pass


def pool2d_vector(a, device):
    #
    # Add your code here
    #
    pass


def relu_scalar(a, device):
    #
    # Add your code here
    #
    pass


def relu_vector(a, device):
    #
    # Add your code here
    #
    pass


def reshape_vector(a, device):
    #
    # Add your code here
    #
    pass


def reshape_scalar(a, device):
    #
    # Add your code here
    #
    pass

def fc_layer_scalar(a, weight, bias, device):
    #
    # Add your code here
    #
    pass


def fc_layer_vector(a, weight, bias, device):
    #
    # Add your code here
    #
    pass
