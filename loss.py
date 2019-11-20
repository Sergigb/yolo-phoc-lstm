import torch.nn as nn
import torch


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        # self.regloss = nn.L1Loss()
        self.regloss = nn.MSELoss()
        self.entropy_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, preds_loc, preds_class, labels, lambda_coord, lambda_noobj):
        masked_bbox = preds_loc[:, :, 0:4] * labels[:, :, 4].unsqueeze(2)
        masked_bbox[masked_bbox == 0] = 1.0  # set the masked values to be 1 to avoid problems during the backprop
        masked_labels = labels[:, :, 0:4] * labels[:, :, 4].unsqueeze(2)
        masked_labels[masked_labels == 0] = 1.0

        # xy_reg_loss = (masked_bbox[:, :, 0:2] - masked_labels[:, :, 0:2]) ** 2.
        # xy_reg_loss = torch.mean(xy_reg_loss)
        # wh_reg_loss = (torch.sqrt(torch.exp(masked_bbox[:, :, 2:4])) - torch.sqrt(torch.exp(masked_labels[:, :, 2:4]))) ** 2.
        # wh_reg_loss = torch.mean(wh_reg_loss)

        obj_loss = self.entropy_loss(preds_class[:, :, 0], labels[:, :, 4])
        # obj_loss = torch.tensor([0]).type(torch.FloatTensor).cuda()

        xy_reg_loss = self.regloss(masked_bbox[:, :, 0:4], masked_labels[:, :, 0:4])
        # wh_reg_loss = self.regloss(torch.sqrt(masked_bbox[:, :, 2:4]),
        #                            torch.sqrt(masked_labels[:, :, 2:4]))

        bbox_loss = xy_reg_loss  # + wh_reg_loss

        # bbox_loss = (self.entropy_loss(masked_bbox[0:2], masked_labels[0:2]) +
        #              self.entropy_loss(masked_bbox[2:4], masked_labels[2:4])) / 2

        return obj_loss + 100*bbox_loss, (obj_loss, 100*bbox_loss, 0)

