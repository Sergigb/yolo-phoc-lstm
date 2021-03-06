import argparse
import os
import sys

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from model import RNN
from data_loader import get_data_loader
from loss import Loss

def main(args):
    print(sys.argv)

    if not os.path.exists('models'):
        os.mkdir('models')

    num_epochs = args.ne
    lr_decay = args.decay
    learning_rate = args.lr

    data_loader = get_data_loader(args.gt_path, args.tensors_path, args.json_labels_path, args.bs)
    model = RNN(lstm_hidden_size=args.hidden_size)
    if torch.cuda.is_available():
        model.cuda()
    model.train()

    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mm)
    if args.rms:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.mm)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model_loss = torch.nn.BCEWithLogitsLoss()
    # model_loss = Loss()

    losses = []
    p = 1
    try:
        for epoch in range(num_epochs):
            if epoch % args.decay_epoch == 0 and epoch > 0:
                learning_rate = learning_rate * lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

            if epoch in (3, 7, 15):
                if epoch == 3:
                    p = 2 / 3
                if epoch == 7:
                    p = 1 / 3
                if epoch == 15:
                    p = 0

            loss_epoch = []
            loss1_epoch = []
            loss2_epoch = []
            for step, (tensors, masks, gt) in enumerate(data_loader):
                if torch.cuda.is_available():
                    tensors = tensors.cuda()
                    masks = masks.cuda()
                    gt = gt.cuda()
                model.zero_grad()

                out, att = model(tensors, masks, gt, p)
                loss1 = model_loss(out, gt)
                # att[:, :-1, :] -> attention produced (location in the next frame) until the last frame -1 (49)
                # gt[:, 1:, :] -> gt from the second frame until the last frame (49)
                loss2 = model_loss(att[:, :-1, :], gt[:, 1:, :])
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()

                loss_epoch.append(loss.cpu().detach().numpy())
                loss1_epoch.append(loss1.cpu().detach().numpy())
                loss2_epoch.append(loss2.cpu().detach().numpy())

                #print('Epoch ' + str(epoch + 1) + '/' + str(num_epochs) + ' - Step ' + str(step + 1) + '/' +
                #      str(len(data_loader)) + ' - Loss: ' + str(float(loss)) + " (Loss1: " + str(float(loss1))
                #       + ", Loss2: " + str(float(loss2)) + ")")
            loss_epoch_mean = np.mean(np.array(loss_epoch))
            loss1_epoch_mean = np.mean(np.array(loss_epoch))
            loss2_epoch_mean = np.mean(np.array(loss_epoch))
            losses.append(loss_epoch_mean)
            print('Total epoch loss: ' + str(loss_epoch_mean) + " (loss1: " + str(loss1_epoch_mean) + ", loss2: " + str(loss2_epoch_mean) + ")")
            if (epoch + 1) % args.save_epoch == 0 and epoch > 0:
                filename = 'model-epoch-' + str(epoch + 1) + '.pth'
                model_path = os.path.join('models/', filename)
                torch.save(model.state_dict(), model_path)
    except KeyboardInterrupt:
        pass

    filename = 'model-epoch-last.pth'
    model_path = os.path.join('models', filename)
    torch.save(model.state_dict(), model_path)
    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensors_path', type=str, default='../tensors/train',
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
    parser.add_argument('-ne', type=int, default=150, help='Number of epochs')
    parser.add_argument('-bs', type=int, default=16, help='Size of the batch')
    parser.add_argument('--decay', type=float, default=0.1,
                        help='Decay of the learning rate')
    parser.add_argument('--decay_epoch', type=int, default=3,
                        help='Indicates the epoch where we want to reduce the learning rate')
    parser.add_argument('--num_descriptors', type=int, default=361,
                        help='Number of descriptors per frame')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Size of the hidden state of the lstm')
    parser.add_argument('--input_size', type=int, default=256,
                        help='Input size to the next step of the lstm')
    parser.add_argument('--seq_length', type=int, default=50,
                        help='Lenght of the sequences')
    parser.add_argument('-rms', action='store_true',
                        help='RMSProp option')
    # not used
    parser.add_argument('--clipping', type=float, default=0., help='Gradient clipping')
    args = parser.parse_args()

    main(args)

# adam default everything 0.040837266 (10)
# adam lr 1e-2 0.0333 (6)