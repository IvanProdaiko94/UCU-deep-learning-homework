from __future__ import print_function
import torch


def pool2d_vector(a, device):
    N_batch, C_in, img_size, _ = a.shape
    result = torch.empty(N_batch, C_in, img_size // 2, img_size // 2)

    print(
        "\nMax pooling layer:\n"
        "\tInput:   A=[{}x{}x{}x{}]\n".format(N_batch, C_in, img_size, img_size),
        "\tOutput:  Z=[{}x{}x{}x{}]".format(N_batch, C_in, img_size // 2, img_size // 2)
    )

    return result.to(device)


def pool2d_scalar(a, device):
    N_batch, C_in, img_size, _ = a.shape
    result = torch.empty(N_batch, C_in, img_size // 2, img_size // 2)

    print(
        "\nMax pooling layer:\n"
        "\tInput:   A=[{}x{}x{}x{}]\n".format(N_batch, C_in, img_size, img_size),
        "\tOutput:  Z=[{}x{}x{}x{}]".format(N_batch, C_in, img_size // 2, img_size // 2)
    )

    for n in range(N_batch):
        for c_in in range(C_in):
            for i in range(img_size // 2):
                for j in range(img_size // 2):
                    double_i = 2*i
                    double_j = 2*j
                    result[n, c_in, i, j] = max(
                        a[n, c_in, double_i, double_j],
                        a[n, c_in, double_i + 1, double_j],
                        a[n, c_in, double_i, double_j + 1],
                        a[n, c_in, double_i + 1, double_j + 1]
                    )

    return result.to(device)
