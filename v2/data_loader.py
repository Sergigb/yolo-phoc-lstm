import os
import json

import torch
import torch.utils.data as data
import numpy as np


class yolo_phoc_dataset(data.Dataset):
    def __init__(self, gt_root, tensors_root, json_labels_path):
        self.gt_root = gt_root
        self.tensors_root = tensors_root
        self.labels_path = json_labels_path

        with open(json_labels_path) as f:
            self.filenames = json.load(f)
        self.keys = list(self.filenames.keys())

    def __getitem__(self, index):
        tensors_fname, masks_fname = self.filenames[self.keys[index]]

        tensors_path = os.path.join(self.tensors_root, tensors_fname)
        masks_path = os.path.join(self.tensors_root, masks_fname)
        gt_path = os.path.join(self.gt_root, self.keys[index])

        tensors = torch.from_numpy(np.load(tensors_path)).type(torch.FloatTensor)
        tensors = tensors.permute(0, 3, 1, 2)
        masks = np.load(masks_path).squeeze()
        masks = torch.from_numpy(masks.reshape(masks.shape[0], -1)).type(torch.FloatTensor)
        gt = torch.from_numpy(np.load(gt_path)).type(torch.FloatTensor)

        return tensors, masks, gt

    def __len__(self):
        return len(self.keys)


def get_data_loader(gt_root, tensors_root, json_labels_path,
                    batch_size, shuffle=True, num_workers=8,):

    dataset = yolo_phoc_dataset(gt_root, tensors_root, json_labels_path)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers)

    return data_loader
