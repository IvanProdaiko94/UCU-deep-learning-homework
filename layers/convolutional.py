from __future__ import print_function
import torch

from .utilities import im2col, convolved_image_size, ax_plus_b_vector, ax_plus_b_scalar


def conv_weight2rows(conv_weight):
    C_out = conv_weight.shape[0]
    return conv_weight.clone().view(C_out, -1)


def conv2d_vector(x_in, conv_weight, conv_bias, device, padding, stride):
    N_batch, C_in, img_size, _ = x_in.shape  # img_size is actually height, but we assume image is square
    C_out, _, K, _ = conv_weight.shape
    out_size = convolved_image_size(img_size, K, padding, stride)

    x_col = im2col(x_in, K, device, stride, padding)
    w_row = conv_weight2rows(conv_weight)

    result = ax_plus_b_vector(x_col, w_row, conv_bias.view(C_out, 1))
    result = result.view(C_out, N_batch, C_in, out_size, out_size).sum(dim=2)
    return result.transpose(0, 1)


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
