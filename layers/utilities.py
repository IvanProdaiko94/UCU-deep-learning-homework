from __future__ import print_function
import torch


def diff_mse(x, y):
    x_vec = x.view(1, -1).squeeze()
    y_vec = y.view(1, -1).squeeze()
    return torch.mean(torch.pow((x_vec - y_vec), 2)).item()


def ax_plus_b_vector(x, weight, bias):
    return weight.mm(x).add(bias)


def ax_plus_b_scalar(x, weight, bias, h, w, number_of_channels, kernel_size):
    result = 0
    for c_in in range(number_of_channels):
        for i in range(kernel_size):
            for j in range(kernel_size):
                result += x[c_in, h + i, w + j] * weight[c_in, i, j]
    return result + bias


def convolved_image_size(size, kernel_size, padding, stride):
    return ((size - kernel_size + 2 * padding) // stride) + 1


def im2col(img, kernel_size, device, stride=1, padding=0):
    N_batch, C_in, img_size, _ = img.shape
    out_size = convolved_image_size(img_size, kernel_size, padding, stride)

    col = torch.zeros((kernel_size, kernel_size, N_batch, C_in, out_size, out_size))

    margin = stride * out_size

    for x in range(kernel_size):
        for y in range(kernel_size):
            col[x, y] = img[:, :, x:x + margin:stride, y:y + margin:stride]

    return col.view(kernel_size*kernel_size, -1).to(device)
