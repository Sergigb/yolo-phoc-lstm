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
            self.training_samples = json.load(f)

    def __getitem__(self, index):
        sample = self.training_samples[index]
        seq_start = sample["seq_start"]
        seq_end = sample["seq_end"]
        video_name = sample["video_name"]
        object_id = sample["object_id"]

        feat_map = np.load(os.path.join(self.tensors_root, "tensor_" + video_name + ".npy"))
        feat_map = feat_map[seq_start:seq_end]
        feat_map = torch.from_numpy(feat_map).type(torch.FloatTensor)
        feat_map = feat_map.squeeze(0).permute(0, 3, 1, 2)
        padding = torch.zeros((100 - feat_map.shape[0], *feat_map.shape[1:]))  # hardcoded, fix
        feat_map = torch.cat((feat_map, padding), dim=0)

        gt = np.load(os.path.join(self.gt_root, video_name + "_" + object_id + ".npy"))
        gt = torch.from_numpy(gt).type(torch.FloatTensor).squeeze(0)

        return feat_map, gt

    def __len__(self):
        return len(self.training_samples)


def get_data_loader(gt_root, tensors_root, batch_size, json_labels_path, shuffle=True, num_workers=8):

    dataset = yolo_phoc_dataset(gt_root, tensors_root, json_labels_path)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers)

    return data_loader
