from __future__ import print_function
import torch


def reshape_vector(a, device):
    N_batch, _, img_size, _ = a.shape
    return a.clone().view(N_batch, -1).to(device)


def reshape_scalar(a, device):
    N_batch, C_in, img_size, _ = a.shape
    out_vec_length = C_in * img_size ** 2
    print(
        "\nReshape layer:\n"
        "\tInput:   A=[{}x{}x{}x{}]\n".format(N_batch, C_in, img_size, img_size),
        "\tOutput:  Z=[{}x{}]".format(N_batch, out_vec_length)
    )

    result = torch.empty(N_batch, out_vec_length)

    for n in range(N_batch):
        for c_in in range(C_in):
            for i in range(img_size):
                for j in range(img_size):
                    result[n, c_in * i * j] = a[n, c_in, i, j]

    return result.to(device)
