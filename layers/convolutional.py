from __future__ import print_function
import torch


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


def conv2d_vector(x_in, conv_weight, conv_bias, device, padding, stride):
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

    return result.to(device)


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
