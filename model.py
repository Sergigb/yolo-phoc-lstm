import torch


class FrameAttention(torch.nn.Module):
    def __init__(self, num_descriptors=10, hidden_size=256, input_size=256, descriptor_size=6):
        """
        :param hidden_size: size of the hidden state
        :param num_descriptors: number of descriptors per frame
        :param input_size: size of the input to the lstm (each descriptor gets linearly projected)
        :param hidden_size: size of the hidden state of the sequence lstm
        :param descriptor_size: size of the descriptors
        """
        super(FrameAttention, self).__init__()

        self.desc_to_input = torch.nn.Linear(descriptor_size, input_size)
        self.w1 = torch.nn.Linear(input_size*num_descriptors+hidden_size, input_size*num_descriptors+hidden_size)
        self.w2 = torch.nn.Linear(input_size*num_descriptors+hidden_size, hidden_size)
        self.v = torch.nn.Linear(input_size*num_descriptors+hidden_size, num_descriptors)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, descriptors, hidden_state):
        """
        :param descriptors: of shape [bs, num_descriptors, descriptor_size]
        :param hidden_state: previous hidden state of the lstm
        :return: attention over the descriptors of the current frame
        """

        embedded = self.desc_to_input(descriptors)
        x = torch.reshape(embedded, (embedded.shape[0], embedded.shape[1]*embedded.shape[2]))
        x = torch.cat((x, hidden_state), dim=-1)
        x = self.relu(self.w1(x))
        next_input = self.relu(self.w2(x))
        a = self.v(x)
        if not self.training:
            a = self.sigmoid(a)

        return a, next_input

class RNN(torch.nn.Module):
    def __init__(self, sequence_length=100, descriptor_size=6, num_descriptors=10, lstm_in_size=256, hidden_size=256):
        super(RNN, self).__init__()

        self.sequence_length = sequence_length
        self.num_descriptors = num_descriptors
        self.descriptor_size = descriptor_size
        self.lstm_in_size = lstm_in_size
        self.hidden_size = hidden_size

        self.frame_attention = FrameAttention(num_descriptors=num_descriptors, descriptor_size=descriptor_size)
        self.lstmcell = torch.nn.LSTMCell(self.lstm_in_size, self.hidden_size)

    def forward(self, descriptors):
        """
        :param descriptors: [bs, sequence length, num descriptors, size descriptors]
        :return: attention mask over the descriptors
        """
        attention = torch.zeros((descriptors.shape[0], self.sequence_length, self.num_descriptors)).cuda()
        h_t = torch.zeros((descriptors.shape[0], self.hidden_size)).cuda()
        c_t = torch.zeros((descriptors.shape[0], self.hidden_size)).cuda()
        input_t = torch.zeros((descriptors.shape[0], self.hidden_size)).cuda()

        for i in range(self.sequence_length):
            h_t, c_t = self.lstmcell(input_t, (h_t, c_t))
            attention[:, i, :], input_t = self.frame_attention(descriptors[:, i, :, :], h_t)

        return attention

