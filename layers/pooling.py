from __future__ import print_function
import torch
from .utilities import im2col


def pool2d_vector(a, device, window_size=2):
    N_batch, C_in, img_size, _ = a.shape
    z_col = im2col(a, window_size, device, stride=2)
    out_size = img_size // window_size
    max_val = z_col.max(dim=0)[0]
    return max_val.view(N_batch, C_in, out_size, out_size)


def pool2d_scalar(a, device, window_size=2):
    N_batch, C_in, img_size, _ = a.shape
    result = torch.empty(N_batch, C_in, img_size // window_size, img_size // window_size)

    print(
        "\nMax pooling layer:\n"
        "\tInput:   A=[{}x{}x{}x{}]\n".format(N_batch, C_in, img_size, img_size),
        "\tOutput:  Z=[{}x{}x{}x{}]".format(N_batch, C_in, img_size // window_size, img_size // window_size)
    )

    for n in range(N_batch):
        for c_in in range(C_in):
            for i in range(img_size // window_size):
                for j in range(img_size // window_size):
                    double_i = window_size*i
                    double_j = window_size*j
                    result[n, c_in, i, j] = max(
                        a[n, c_in, double_i, double_j],
                        a[n, c_in, double_i + 1, double_j],
                        a[n, c_in, double_i, double_j + 1],
                        a[n, c_in, double_i + 1, double_j + 1]
                    )

    return result.to(device)
