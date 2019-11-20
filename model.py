import torch.nn as nn
import torch


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size=1024):
        """
        :param input_size:
        :param hidden_size:
        :param out_size:
        """
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=1)
        self.linear_classification = nn.Linear(hidden_size, 1)
        # self.linear_obj1 = nn.Linear(hidden_size, 1)
        # self.linear_obj2 = nn.Linear(512, 1)
        self.linear_bbox1 = nn.Linear(hidden_size, 2046)
        self.linear_bbox2 = nn.Linear(2046, 2048)
        self.linear_bbox3 = nn.Linear(2048, 4)
        # self.linear_bbox4 = nn.Linear(512, 4)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, descriptors):
        hiddens, _ = self.lstm(descriptors)
        outputs = self.relu(hiddens)
        preds_obj = self.linear_classification(outputs)
        # preds_obj = self.linear_obj1(outputs)
        # preds_obj = self.relu(preds_obj)
        # preds_obj = self.linear_obj2(preds_obj)

        preds_loc = self.linear_bbox1(outputs)
        preds_loc = self.relu(preds_loc)
        # preds_loc = self.dropout(preds_loc)
        preds_loc = self.linear_bbox2(preds_loc)
        preds_loc = self.relu(preds_loc)
        # preds_loc = self.dropout(preds_loc)
        preds_loc = self.linear_bbox3(preds_loc)
        # preds_loc = self.relu(preds_loc)
        # preds_loc = self.dropout(preds_loc)
        # preds_loc = self.linear_bbox4(preds_loc)
        preds_loc = self.sigm(preds_loc)

        if not self.training:
            preds_obj[:, :, -1] = torch.sigmoid(preds_obj[:, :, -1])
            # preds_loc = torch.sigmoid(preds_loc)  # if the loss is cross entropy

        return preds_loc, preds_obj

