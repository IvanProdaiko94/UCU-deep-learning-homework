import torch


def fc_layer_vector(a, weight, bias, device):
    N_batch = a.shape[0]
    D, P = weight.shape

    print(
        "\nConvolution layer:\n"
        "\tInput:   A=[{}x{}]\n".format(N_batch, P),
        "\tWeights: W=[{}x{}]\n".format(D, P),
        "\tBias:    B=[{}]\n".format(bias.shape[0]),
        "\tOutput:  Z=[{}x{}]".format(N_batch, P)
    )

    result = torch.empty(N_batch, D)

    return result.to(device)


def fc_layer_scalar(a, weight, bias, device):
    N_batch = a.shape[0]
    D, P = weight.shape

    print(
        "\nConvolution layer:\n"
        "\tInput:   A=[{}x{}]\n".format(N_batch, P),
        "\tWeights: W=[{}x{}]\n".format(D, P),
        "\tBias:    B=[{}]\n".format(bias.shape[0]),
        "\tOutput:  Z=[{}x{}]".format(N_batch, P)
    )

    result = torch.empty(N_batch, D)

    for n in range(N_batch):
        for j in range(D):
            for i in range(P):
                result[n, j] += weight[j, i] * a[n, i] + bias[j]

    return result.to(device)
