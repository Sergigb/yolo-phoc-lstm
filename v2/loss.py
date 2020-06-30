import torch


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.bceloss = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, predicted, gt, lambd=7):
        # loss = torch.zeros(1).cuda()
        # really slow, there must be a faster way
        # for i in range(predicted.shape[0]):
        #     for j in range(predicted.shape[1]):
        #         if torch.sum(gt[i, j, :]) > 0:
        #             loss += self.bceloss(predicted[i, j, :], gt[i, j, :]) * lambd
        #         else:
        #             loss += self.bceloss(predicted[i, j, :], gt[i, j, :])
        # loss /= (predicted.shape[0] * predicted.shape[1])
        #
        # print("slow method: ", loss)

        bcel = self.bceloss(predicted, gt)
        bcel2 = bcel * lambd

        loss2 = torch.mean(torch.where(gt > 0, bcel2, bcel))

        # print("fast? method: ", loss2)



        return loss2

