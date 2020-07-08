import torch
import sys


class RNN(torch.nn.Module):
    def __init__(self, lstm_in_size=93860, lstm_hidden_size=1024, sequence_length=50, att_size=38):
        super(RNN, self).__init__()
        #self.w2 = torch.nn.Linear(lstm_hidden_size, 5)
        self.att_size = att_size
        self.hidden_size = lstm_hidden_size
        self.sequence_length = sequence_length

        self.lstmcell = torch.nn.LSTMCell(lstm_in_size, lstm_hidden_size)
        self.h_to_att = torch.nn.Linear(lstm_hidden_size, att_size * att_size)
        self.h_to_out = torch.nn.Linear(lstm_hidden_size, att_size * att_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()


    def forward(self, tensors, masks):
        """
        
        """

        h_t = torch.zeros((tensors.shape[0], self.hidden_size)).cuda()
        c_t = torch.zeros((tensors.shape[0], self.hidden_size)).cuda()
        conv_att_t = torch.zeros((tensors.shape[0], self.att_size, self.att_size)).cuda()  # we only keep current one for now
        out = torch.zeros((tensors.shape[0], self.sequence_length, self.att_size * self.att_size)).cuda()

        for i in range(self.sequence_length):

            tensors_slice_t = torch.zeros((tensors.shape[0], *tensors.shape[-3:])).cuda()
            for j in range(tensors.shape[0]):
                tensors_slice_t[j] = tensors[j, i, :] * conv_att_t[j, :].unsqueeze(-1)  # apply attention

            input_tensor = tensors_slice_t.reshape(tensors_slice_t.shape[0], -1)
            input_mask = masks[:, i, :]
            input_t = torch.cat((input_tensor, input_mask), dim=1)
            #input_t = input_mask

            h_t, c_t = self.lstmcell(input_t, (h_t, c_t))

            h_t_ReLU = self.relu(h_t)
            conv_att_t = self.h_to_att(h_t_ReLU)
            conv_att_t = self.sigmoid(conv_att_t)
            conv_att_t = conv_att_t.reshape(conv_att_t.shape[0], self.att_size, self.att_size)


            out[:, i, :] = self.h_to_out(h_t_ReLU)

        if not self.training:
            out = self.sigmoid(out)

        return out

