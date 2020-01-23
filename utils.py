import xml.etree.cElementTree as ET
from glob import glob
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from model import RNN

trans = str.maketrans({'.': r'', '"': r'', '\n': r'', '-': r'', '\'': r''})

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
    gt_bboxes = [frame1, frame2, ..., frameN]
    frameN = [annotation1, annotation2, ..., annotationN]
    annotationN = [x, y, w, h]
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
    def __init__(self, input_size=600, hidden_size=256, weights_path='models/best/model-epoch-last.pth',
                 num_descriptors=10):
        self.model = RNN(num_descriptors=num_descriptors, hidden_size=hidden_size, lstm_in_size=input_size)
        self.model.load_state_dict(torch.load(weights_path))
        self.num_descriptors = num_descriptors

        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def sample_video(self, query, video_name, descriptors_path='extracted_descriptors_100', print_sorted_files=False):
        self.model.eval()

        files = glob(os.path.join(descriptors_path, 'descriptors_top'+ str(self.num_descriptors) + '_' + video_name +
                                  '_' + query + '_*'))
        files = sorted(files)
        if print_sorted_files:
            print(os.path.join(descriptors_path, 'descriptors_top' + str(self.num_descriptors) + '_' + video_name +
                               '_' + query + '_*'))
            print(files)

        predictions = None
        for desc_file in files:
            descriptors = np.load(desc_file)
            descriptors = torch.from_numpy(descriptors).type(torch.FloatTensor)\
                .reshape((1, descriptors.shape[1], int(descriptors.shape[2]/6), 6))
            if torch.cuda.is_available():
                descriptors = descriptors.cuda()
            preds = self.model(descriptors)
            if predictions is None:
                predictions = preds
            else:
                predictions = torch.cat((predictions, preds), 1)

        return predictions


def iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    bbox1area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    iou_score = intersection / float((bbox1area + bbox2area) - intersection)
    return iou_score


def load_descriptors(video_name, query, descriptors_path, num_descriptors=10):
    files = glob(os.path.join(descriptors_path, 'descriptors_top' + str(num_descriptors) + '_' + video_name + '_' +
                              query + '_*'))
    files = sorted(files)
    descriptors = None
    for file in files:
        if descriptors is None:
            descriptors = np.load(file).squeeze()
        else:
            descriptors = np.concatenate((descriptors, np.load(file).squeeze()))
    descriptors = descriptors.reshape((descriptors.shape[0], int(descriptors.shape[1]/6), 6))
    return descriptors






