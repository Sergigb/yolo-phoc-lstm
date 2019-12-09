import os
import json

import torch
import torch.utils.data as data
import numpy as np


class yolo_phoc_dataset(data.Dataset):
    def __init__(self, gt_root, descriptors_path, json_labels_path):
        self.gt_root = gt_root
        self.descriptors_path = descriptors_path
        self.json_labels_path = json_labels_path

        with open(json_labels_path) as f:
            self.labels = json.load(f)
        self.labels_keys = list(self.labels.keys())

        self.gt = {}
        for key in self.labels.keys():
            path = os.path.join(self.gt_root, key)
            self.gt[key] = torch.from_numpy(np.load(path)).type(torch.FloatTensor)

    def __getitem__(self, index):
        descriptor_fname = self.labels[self.labels_keys[index]]

        gt = self.gt[self.labels_keys[index]]
        descriptor = np.load(os.path.join(self.descriptors_path, descriptor_fname))
        shape = descriptor.shape
        descriptor = descriptor.reshape((shape[0], shape[1], int(shape[2]/6), 6))
        descriptor = torch.from_numpy(descriptor).type(torch.FloatTensor).squeeze(0)

        return descriptor, gt

    def __len__(self):
        return len(self.labels_keys)


def get_data_loader(gt_root, descriptors_path, json_labels_path,
                    batch_size, shuffle=True, num_workers=8,):

    dataset = yolo_phoc_dataset(gt_root, descriptors_path, json_labels_path)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers)

    return data_loader
