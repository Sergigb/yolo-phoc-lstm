from random import random

import torch
import sys


class RNN(torch.nn.Module):
    def __init__(self, lstm_in_size=12996, lstm_hidden_size=512, sequence_length=50, att_size=38):
        super(RNN, self).__init__()
        self.att_size = att_size
        self.hidden_size = lstm_hidden_size
        self.sequence_length = sequence_length

        self.conv = torch.nn.Conv2d(64, 8, 3, padding=1)
        self.lstmcell = torch.nn.LSTMCell(lstm_in_size, lstm_hidden_size)
        self.h_to_att = torch.nn.Linear(lstm_hidden_size, att_size * att_size)
        self.h_to_out = torch.nn.Linear(lstm_hidden_size, att_size * att_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()


    def forward(self, tensors, masks, gt=None, p=0):
        """
        idea: make a new layer with h_t + masks + ?features?
        """
        h_t = torch.zeros((tensors.shape[0], self.hidden_size)).cuda()
        c_t = torch.zeros((tensors.shape[0], self.hidden_size)).cuda()
        conv_att_t = torch.zeros((tensors.shape[0], self.att_size, self.att_size)).cuda()
        out = torch.zeros((tensors.shape[0], self.sequence_length, self.att_size * self.att_size)).cuda()
        conv_att = torch.zeros((tensors.shape[0], self.sequence_length, self.att_size * self.att_size)).cuda()

        for i in range(self.sequence_length):
            tensors_conv_t = self.conv(tensors[:, i, :])
            tensors_att_t = torch.zeros((tensors_conv_t.shape[0], * tensors_conv_t.shape[-3:])).cuda()

            for j in range(tensors.shape[0]):
                if not p:
                    tensors_att_t[j] = tensors_conv_t[j] * conv_att_t[j, :].unsqueeze(0)
                else:
                    if random() >= p:
                        tensors_att_t[j] = tensors_conv_t[j] * conv_att_t[j, :].unsqueeze(0)
                    else:
                        tensors_att_t[j] = tensors_conv_t[j] * gt[j, i, :].reshape(1, self.att_size, self.att_size)

            input_tensor = tensors_att_t.reshape(tensors_att_t.shape[0], -1)
            input_mask = masks[:, i, :]

            input_t = torch.cat((input_tensor, input_mask), dim=1)

            h_t, c_t = self.lstmcell(input_t, (h_t, c_t))

            h_t_ReLU = self.relu(h_t)
            conv_att_t = self.h_to_att(h_t_ReLU)
            conv_att[:, i, :] = conv_att_t
            conv_att_t = self.sigmoid(conv_att_t)
            conv_att_t = conv_att_t.reshape(conv_att_t.shape[0], self.att_size, self.att_size)

            out[:, i, :] = self.h_to_out(h_t_ReLU)

        if not self.training:
            out = self.sigmoid(out)
            conv_att = self.sigmoid(conv_att)


        return out, conv_att

