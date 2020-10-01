from random import random

import torch
import sys



class ConvLSTMCell(torch.nn.Module):
    # from https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = torch.nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                    out_channels=4 * self.hidden_dim,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class RNN(torch.nn.Module):
    def __init__(self, ):
        super(RNN, self).__init__()
        self.in_size = 32
        self.convlstm = ConvLSTMCell(self.in_size, self.in_size, (3, 3), bias=False)
        self.feats_to_in = torch.nn.Conv2d(in_channels=64, out_channels=self.in_size // 2, kernel_size=(3, 3), padding=True)
        self.mask_to_in = torch.nn.Conv2d(in_channels=1, out_channels=self.in_size // 2, kernel_size=(3, 3), padding=True)
        self.out_to_mask = torch.nn.Conv2d(in_channels=self.in_size, out_channels=1, kernel_size=(3, 3), padding=True)
        self.relu = torch.nn.ReLU()

    def forward(self, feat_maps, gt, p=1.0):
        """
        
        """
        masks = torch.zeros(feat_maps.shape[0], feat_maps.shape[1], 38, 38).cuda()
        h_t = torch.zeros(feat_maps.shape[0], self.in_size, 38, 38).cuda()
        c_t = torch.zeros(feat_maps.shape[0], self.in_size, 38, 38).cuda()

        masks[:, 0] = gt[:, 0]

        for i in range(1, feat_maps.shape[1]):
            feats_in = self.feats_to_in(feat_maps[:, i])
            if(random() > p):
                mask_in = self.mask_to_in(masks[:, i-1].unsqueeze(1))
            else:
                mask_in = self.mask_to_in(gt[:, i-1].unsqueeze(1))

            in_t = torch.cat((feats_in, mask_in), dim=1)

            h_t, c_t = self.convlstm(in_t, (h_t, c_t))
            mask = self.out_to_mask(h_t)
            masks[:, i] = mask.squeeze(1)

        if not self.training:
            masks = torch.sigmoid(masks)

        return masks

