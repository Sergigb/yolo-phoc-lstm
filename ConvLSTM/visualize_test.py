import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import json



feat_map_path = "../tensors/train/tensor_Video_41_2_3.npy"
gt_path = "gt/Video_41_2_3_49001.npy"


gt_samples_path = "gt/gt.json"
with open(gt_samples_path) as f:
    training_samples = json.load(f)

sample = None
for item in training_samples:
    if item["video_name"] == "Video_41_2_3" and item["object_id"] == "49001":
        sample = item
        break
seq_start = sample["seq_start"]
seq_end = sample["seq_end"]


feat_map = np.load(feat_map_path)
print(feat_map.shape)


gt = np.load(gt_path)
feat_map[seq_start] = feat_map[seq_start] * gt[0].reshape((*gt[0].shape, 1))
feat_map = np.sum(feat_map, axis=3)
feat_map = feat_map / np.linalg.norm(feat_map)

cap = cv2.VideoCapture("../datasets/rrc-text-videos/ch3_train/Video_41_2_3.mp4")
video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_POS_FRAMES, float(seq_start));


fig, ax = plt.subplots(1)
ret, inp = cap.read()
current_frame = seq_start
while ret:
    im = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    plt.cla()
    ax.imshow(im)

    feat = cv2.resize(feat_map[current_frame] / 1, (int(video_width), int(video_height)), interpolation = cv2.INTER_AREA)
    ax.imshow(feat, cmap='jet', alpha=0.5)

    plt.axis('off')
    plt.show(block=False)
    plt.pause(1.00001)

    ret, inp = cap.read()
    current_frame += 1

plt.close()
cap.release()



















