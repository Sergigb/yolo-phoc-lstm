import xml.etree.cElementTree as ET
from glob import glob
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from model import RNN


def plot_grad_flow(named_parameters):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        print(n)
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    plt.show()


def get_gt_from_file(gtfile):
    """
    gt = [frame1, frame2, ..., frameN]
    frameN = [annotation1, annotation2, ..., annotationN]
    annotationN = [object_id, [x, y, w, h]]
    :param gtfile: path to ground truth file
    :return: gt
    """
    tree = ET.parse(gtfile)
    root = tree.getroot()

    gt_bboxes = []
    gt_ids = []
    for frame in root:
        bboxes_frame = []
        ids_frame = []
        for object_ in frame:
            if object_.get('Quality') in ('HIGH', 'MODERATE'):
                ids_frame.append(object_.get('ID'))

                x_coords = []
                y_coords = []
                for point in object_.iter('Point'):
                    x_coords.append(int(point.get('x')))
                    y_coords.append(int(point.get('y')))
                w = max(x_coords) - min(x_coords)
                h = max(y_coords) - min(y_coords)
                x = min(x_coords)
                y = min(y_coords)
                bboxes_frame.append([x, y, w, h])

        gt_bboxes.append(bboxes_frame)
        gt_ids.append(ids_frame)
    return gt_bboxes, gt_ids


class Sampler:
    """
    Samples all the detections for a given video and query
    """
    def __init__(self, input_size=600, hidden_size=1024, weights_path='models/best/model-epoch-last.pth'):
        self.model = RNN(input_size, hidden_size=hidden_size)
        self.model.load_state_dict(torch.load(weights_path))

    def sample_video(self, query, video_name, descriptors_path='extracted_descriptors_100', print_sorted_files=False):
        self.model.eval()

        files = glob(os.path.join(descriptors_path, 'descriptors_top100_' + video_name + '_' +
                                       query + '_*'))
        files = sorted(files)
        if print_sorted_files:
            print(os.path.join(descriptors_path, 'descriptors_top100_' + video_name + '_' + query + '_*'))
            print(files)

        predictions = None
        predictions_loc = None
        for desc_file in files:
            descriptors = np.load(desc_file)
            descriptors = torch.from_numpy(descriptors).type(torch.FloatTensor)
            if torch.cuda.is_available():
                self.model.cuda()
                descriptors = descriptors.cuda()
            self.model.eval()
            preds_loc, preds = self.model(descriptors)
            if predictions is None:
                predictions = preds
                predictions_loc = preds_loc
            else:
                predictions = torch.cat((predictions, preds), 1)
                predictions_loc = torch.cat((predictions_loc, preds_loc), 1)

        return predictions_loc, predictions














