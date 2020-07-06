import os
import json

import torch
import torch.utils.data as data
import numpy as np


class yolo_phoc_dataset(data.Dataset):
    def __init__(self, gt_root, tensors_path, json_labels_path):
        self.gt_root = gt_root
        self.tensors_path = tensors_path
        self.json_labels_path = json_labels_path

        with open(json_labels_path) as f:
            self.tensors_fnames = json.load(f)
        self.tensor_keys = list(self.tensors_fnames.keys())

    def __getitem__(self, index):
        tensor_fname, descriptor_fname, phoc_path = self.tensors_fnames[self.tensor_keys[index]]
        key_path = os.path.join(self.gt_root, self.tensor_keys[index])

        gt = torch.from_numpy(np.load(key_path)).type(torch.FloatTensor)
        tensor = np.load(os.path.join(self.tensors_path, tensor_fname))
        #tensor = tensor.swapaxes(1,3).swapaxes(2,3)  # not sure if this is right
        #tensor = torch.from_numpy(tensor).type(torch.FloatTensor)

        #phoc = np.load(phoc_path)
        descriptor =  np.load(os.path.join(self.tensors_path, descriptor_fname))

        #new_descriptor = []
        #for i in range(descriptor.shape[0]):
        #    new_descriptor.append(np.concatenate((descriptor[i], phoc.squeeze(0))))
        #descriptor = torch.from_numpy(np.array(new_descriptor)).type(torch.FloatTensor)

        return tensor, descriptor, gt

    def __len__(self):
        return len(self.tensor_keys)


def get_data_loader(gt_root, tensors_path, json_labels_path,
                    batch_size, shuffle=True, num_workers=8,):

    dataset = yolo_phoc_dataset(gt_root, tensors_path, json_labels_path)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers)

    return data_loader
