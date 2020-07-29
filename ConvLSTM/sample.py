import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from model import RNN
import json

object_id = "17002"
video_name = "Video_37_2_3"
model_path = "models/model-epoch-50.pth"

feat_map_path = "../tensors/train/tensor_" + video_name + ".npy"
gt_path = "gt/" + video_name + "_" + object_id + ".npy"

feat_map = np.load(feat_map_path)
gt = np.load(gt_path)
gt = torch.from_numpy(gt).type(torch.FloatTensor).squeeze(0)


gt_samples_path = "gt/gt.json"
with open(gt_samples_path) as f:
    training_samples = json.load(f)

sample = None
for item in training_samples:
    if item["video_name"] == video_name and item["object_id"] == object_id:
        sample = item
        break

seq_start = sample["seq_start"]
seq_end = sample["seq_end"]

feat_map = feat_map[seq_start:seq_end]
feat_map = torch.from_numpy(feat_map).type(torch.FloatTensor)
feat_map = feat_map.squeeze(0).permute(0, 3, 1, 2)
padding = torch.zeros((100 - feat_map.shape[0], *feat_map.shape[1:]))
feat_map = torch.cat((feat_map, padding), dim=0)

feat_map = feat_map.unsqueeze(0).cuda()
gt = gt.unsqueeze(0).cuda()

model = RNN()
model.load_state_dict(torch.load(model_path))
model.cuda()
model.eval()

out = model(feat_map, gt)
out = out.cpu().detach().numpy()
print(out.shape)

cap = cv2.VideoCapture("../datasets/rrc-text-videos/ch3_train/" + video_name + ".mp4")
video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_POS_FRAMES, float(seq_start));

fig, ax = plt.subplots(1)
ret, inp = cap.read()
current_frame = 0
for i in range(seq_end - seq_start):
    im = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    plt.cla()
    ax.imshow(im)

    heat = cv2.resize(out[0, current_frame] / 1, (int(video_width), int(video_height)), interpolation = cv2.INTER_AREA)
    ax.imshow(heat, cmap='jet', alpha=0.5)

    plt.axis('off')
    #plt.show(block=False)
    #plt.pause(0.00001)

    plt.savefig('images/file%05d.jpeg' % current_frame, bbox_inches = 'tight', pad_inches = 0)

    ret, inp = cap.read()
    current_frame += 1

plt.close()
cap.release()

os.system("ffmpeg -framerate 10 -pattern_type glob -i 'images/*.jpeg' -c:v mpeg4 -vb 1M -qscale:v 2 " + video_name + ".mp4")



