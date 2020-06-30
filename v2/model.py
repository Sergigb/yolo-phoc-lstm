import torch


class RNN(torch.nn.Module):
    def __init__(self, num_descriptors=361, descriptor_size=6, lstm_in_size=29845, lstm_hidden_size=1024, sequence_length=50):
        super(RNN, self).__init__()
        self.w2 = torch.nn.Linear(lstm_hidden_size, 5)
        self.lstmcell = torch.nn.LSTMCell(lstm_in_size, lstm_hidden_size)
        self.hidden_size = lstm_hidden_size
        self.sequence_length = sequence_length
        self.sigmoid = torch.nn.Sigmoid()


        self.conv = torch.nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=2, padding=1)

    def forward(self, tensors, descriptors):
        """
        :param descriptors: [bs, sequence length, num descriptors, size descriptors]
        :return: attention mask over the descriptors
        """

        h_t = torch.zeros((descriptors.shape[0], self.hidden_size)).cuda()
        c_t = torch.zeros((descriptors.shape[0], self.hidden_size)).cuda()
        out = torch.zeros((descriptors.shape[0], self.sequence_length, 5)).cuda()

        for i in range(self.sequence_length):
            visual_features = self.conv(tensors[:, i])
            visual_features = visual_features.reshape((visual_features.shape[0], -1))
            input_t = torch.cat((descriptors[:, i], visual_features), dim=1)

            h_t, c_t = self.lstmcell(input_t, (h_t, c_t))
            out[:, i, :] = self.w2(h_t)

        if not self.training:
            print("sigmoiding")
            a = self.sigmoid(out)

        return out

