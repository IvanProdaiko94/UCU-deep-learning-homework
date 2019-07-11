from __future__ import print_function
import torch


def diff_mse(x, y):
    x_vec = x.view(1, -1).squeeze()
    y_vec = y.view(1, -1).squeeze()
    return torch.mean(torch.pow((x_vec - y_vec), 2)).item()


def convolved_image_size(size, kernel_size, padding, stride):
    return ((size - kernel_size + 2*padding) // stride) + 1


def ax_plus_b_vector():
    pass


def ax_plus_b_scalar(x, weight, bias, h, w, number_of_channels, kernel_size):
    result = 0
    for c_in in range(number_of_channels):
        for i in range(kernel_size):
            for j in range(kernel_size):
                result += x[c_in, h + i, w + j] * weight[c_in, i, j]
    return result + bias

