from __future__ import print_function
import torch


def relu_vector(a, device):
    result = a.clone()
    result[result < 0] = 0
    return result


def relu_scalar(a, device):
    N_batch, L = a.shape
    print(
        "\nRelu layer:\n"
        "\tInput:   A=[{}x{}]\n".format(N_batch, L),
        "\tOutput:  Z=[{}x{}]".format(N_batch, L)
    )

    result = torch.empty((N_batch, L))

    for n in range(N_batch):
        for i in range(L):
            result[n, i] = max(0, a[n, i])

    return result.to(device)
