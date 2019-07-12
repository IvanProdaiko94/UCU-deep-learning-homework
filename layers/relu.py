from __future__ import print_function
import torch


def relu_scalar(a, device):
    N_batch, L = a.shape
    print(
        "\nConvolution layer:\n"
        "\tInput:   A=[{}x{}]\n".format(N_batch, L),
        "\tOutput:  Z=[{}x{}]".format(N_batch, L)
    )

    result = torch.empty((N_batch, L))

    for n in range(N_batch):
        for i in range(L):
            result[n, i] = max(0, a[n, i])

    return result.to(device)


def relu_vector(a, device):
    N_batch, L = a.shape
    print(
        "\nConvolution layer:\n"
        "\tInput:   A=[{}x{}]\n".format(N_batch, L),
        "\tOutput:  Z=[{}x{}]".format(N_batch, L)
    )

    result = torch.empty((N_batch, L))

    return result.to(device)
