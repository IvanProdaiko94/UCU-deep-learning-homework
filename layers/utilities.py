from __future__ import print_function
import torch


def diff_mse(x, y):
    x_vec = x.view(1, -1).squeeze()
    y_vec = y.view(1, -1).squeeze()
    return torch.mean(torch.pow((x_vec - y_vec), 2)).item()


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
