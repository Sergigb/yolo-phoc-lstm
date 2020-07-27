import os

import numpy as np
import cv2
import matplotlib.pyplot as plt



feat_map_path = "../tensors/train/tensor_Video_41_2_3.npy"
gt_path = "gt/Video_41_2_3_1001.npy"

feat_map = np.load(feat_map_path)
feat_map = np.sum(feat_map, axis=3)
feat_map = feat_map / np.linalg.norm(feat_map)
print(feat_map.shape)

gt = np.load(gt_path)


cap = cv2.VideoCapture("../datasets/rrc-text-videos/ch3_train/Video_41_2_3.mp4")
video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


fig, ax = plt.subplots(1)
ret, inp = cap.read()
current_frame = 0
while ret:
    im = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    plt.cla()
    ax.imshow(im)

    feat = cv2.resize(feat_map[current_frame], (int(video_width), int(video_height)), interpolation = cv2.INTER_AREA)
    ax.imshow(feat, cmap='jet', alpha=0.5)

    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.00001)

    ret, inp = cap.read()
    current_frame += 1

plt.close()
cap.release()



















