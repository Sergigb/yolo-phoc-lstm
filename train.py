import argparse
import os

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from model import RNN
from data_loader import get_data_loader
from loss import Loss
from utils import plot_grad_flow


def main(args):
    if not os.path.exists('models'):
        os.mkdir('models')

    num_epochs = args.ne
    lr_decay = args.decay
    learning_rate = args.lr

    data_loader = get_data_loader(args.gt_path, args.descriptors_path, args.json_labels_path, args.bs)
    model = RNN(input_size=600, hidden_size=1024)  # ... TODO: change hardcoding the number of descriptors
    if torch.cuda.is_available():
        model.cuda()
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mm)
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.mm)
    model_loss = Loss()

    losses = []
    for epoch in range(num_epochs):
        if epoch % args.decay_epoch == 0 and epoch > 0:
            learning_rate = learning_rate * lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        loss_epoch = []
        loss_epoch_loc = []
        loss_epoch_obj = []
        for step, (descriptors, labels) in enumerate(data_loader):
            if torch.cuda.is_available():
                descriptors = descriptors.cuda()
                labels = labels.cuda()
            model.zero_grad()

            preds_loc, preds_obj = model(descriptors)
            loss, partial_losses = model_loss(preds_loc, preds_obj, labels, 0, 0)
            loss.backward()
            optimizer.step()

            loss_epoch.append(loss.cpu().detach().numpy())
            loss_epoch_obj.append(partial_losses[0].cpu().detach().numpy())
            loss_epoch_loc.append(partial_losses[1].cpu().detach().numpy())

            print('Epoch ' + str(epoch + 1) + '/' + str(num_epochs) + ' - Step ' + str(step + 1) + '/' +
                  str(len(data_loader)) + ' - Loss: ' + str(float(loss)) + ' (' + str(partial_losses[0].item()) + ', '
                  + str(partial_losses[1].item()) + ', ' + str(partial_losses[2]) + ')')

        loss_epoch_mean = np.mean(np.array(loss_epoch))
        losses.append(loss_epoch_mean)
        print('Total epoch loss: ' + str(loss_epoch_mean)
              + ' - Loc loss: ' + str(np.mean(np.array(loss_epoch_loc)))
              + ' - Obj loss: ' + str(np.mean(np.array(loss_epoch_obj))))
        if (epoch + 1) % args.save_epoch == 0 and epoch > 0:
            filename = 'model-epoch-' + str(epoch + 1) + '.pth'
            model_path = os.path.join('models', filename)
            torch.save(model.state_dict(), model_path)

    plt.plot(losses)
    plt.show()
    filename = 'model-epoch-last' + '.pth'
    model_path = os.path.join('models', filename)
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--descriptors_path', type=str, default='extracted_descriptors/',
                        help='Path to the descriptors')
    parser.add_argument('--gt_path', type=str, default='gt', help='Path to the ground truth labels')
    parser.add_argument('--json_labels_path', type=str, default='gt/labels.json',
                        help='Path to the json with the labels')
    parser.add_argument('--n_workers', type=int, default=8,
                        help='Number of subprocesses used for data loading')
    parser.add_argument('-lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('-mm', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--save_epoch', type=int, default=2,
                        help='Epoch where we want our model to be saved')
    parser.add_argument('-ne', type=int, default=20, help='Number of epochs')
    parser.add_argument('-bs', type=int, default=16, help='Size of the batch')
    parser.add_argument('--decay', type=float, default=0.1,
                        help='Decay of the learning rate')
    parser.add_argument('--decay_epoch', type=int, default=3,
                        help='Indicates the epoch where we want to reduce the learning rate')
    parser.add_argument('--max_sequence_length', type=str, default=250,
                        help='Maximum length of each one of the sequences')
    # not used
    parser.add_argument('--clipping', type=float, default=0., help='Gradient clipping')
    args = parser.parse_args()

    main(args)
