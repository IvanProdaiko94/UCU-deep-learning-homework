from __future__ import print_function
import torch

from utilities import convolved_image_size, ax_plus_b_vector, ax_plus_b_scalar


def conv2d_scalar(x_in, conv_weight, conv_bias, device, padding, stride):
    N_batch, C_in, img_size, _ = x_in.shape  # img_size is actually height, but we assume image is square
    C_out, _, K, _ = conv_weight.shape
    out_size = convolved_image_size(img_size, K, padding, stride)
    print(
        "\nConvolution layer:\n"
        "\tInput:   X=[{}x{}x{}x{}]\n".format(N_batch, C_in, img_size, img_size),
        "\tWeights: W=[{}x{}x{}]\n".format(C_out, K, K),
        "\tBias:    B=[{}]\n".format(conv_bias.shape[0]),
        "\tOutput:  Z=[{}x{}x{}x{}]".format(N_batch, C_out, out_size, out_size)
    )

    result = torch.empty(N_batch, C_out, out_size, out_size)

    for n in range(N_batch):
        for c_out in range(C_out):
            for h in range(out_size):
                for w in range(out_size):
                    weight = conv_weight[c_out]
                    bias = conv_bias[c_out]
                    z = ax_plus_b_scalar(x_in[n], weight, bias, h, w, C_in, K)
                    result[n, c_out, h, w] = z

    return result.to(device)


def conv2d_vector(x_in, conv_weight, conv_bias, device, padding, stride):
    N_batch, C_in, Height, Weight = x_in.shape

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


def pool2d_vector(a, device):
    #
    # Add your code here
    #
    pass


def relu_scalar(a, device):
    N_batch, L = a.shape
    print(
        "\nConvolution layer:\n"
        "\tInput:   A=[{}x{}]\n".format(N_batch, L),
        "\tOutput:  Z=[{}x{}]".format(N_batch, L)
    )

    result = torch.empty((N_batch, L)).to(device)

    for n in range(N_batch):
        for i in range(L):
            result[n, i] = max(0, a[n, i])

    return result


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


def fc_layer_vector(a, weight, bias, device):
    #
    # Add your code here
    #
    pass


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

    z = torch.empty(N_batch, D).to(device)

    for n in range(N_batch):
        for j in range(D):
            for i in range(P):
                z[n, j] += weight[j, i] * a[n, i] + bias[j]

    return z.to(device)
