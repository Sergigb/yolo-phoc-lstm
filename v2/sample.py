import glob
import os
import numpy as np
import torch
from model import RNN
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pat

weights_path = "models/model-epoch-22.pth"
#weights_path = "models/model-epoch-20.pth"

model = RNN(lstm_hidden_size=1024)
model.cuda()
model.eval()
model.load_state_dict(torch.load(weights_path))

video_path = "../datasets/rrc-text-videos/ch3_train/Video_45_6_4.mp4"
descriptor_mask = [torch.from_numpy(np.load("../tensors/train/mask_Video_45_6_4_caprabo_000151.npy")).type(torch.FloatTensor).cuda(),\
                   torch.from_numpy(np.load("../tensors/train/mask_Video_45_6_4_caprabo_000201.npy")).type(torch.FloatTensor).cuda()]

tensors = [torch.from_numpy(np.load("../tensors/train/tensor_Video_45_6_4_000151.npy")).type(torch.FloatTensor).permute(0, 3, 1, 2).unsqueeze(0).cuda(),
           torch.from_numpy(np.load("../tensors/train/tensor_Video_45_6_4_000201.npy")).type(torch.FloatTensor).permute(0, 3, 1, 2).unsqueeze(0).cuda()]

"""video_path = "../datasets/rrc-text-videos/ch3_train/Video_10_1_1.mp4"
descriptor_mask = [torch.from_numpy(np.load("../tensors/train/mask_Video_10_1_1_veterinaria_000051.npy")).type(torch.FloatTensor).cuda(),\
                   torch.from_numpy(np.load("../tensors/train/mask_Video_10_1_1_veterinaria_000101.npy")).type(torch.FloatTensor).cuda()]

tensors = [torch.from_numpy(np.load("../tensors/train/tensor_Video_10_1_1_000051.npy")).type(torch.FloatTensor).permute(0, 3, 1, 2).unsqueeze(0).cuda(),
           torch.from_numpy(np.load("../tensors/train/tensor_Video_10_1_1_000101.npy")).type(torch.FloatTensor).permute(0, 3, 1, 2).unsqueeze(0).cuda()]"""

video_name = video_path.split("/")[-1].replace(".mp4", "")
mask_size = 38

for i, mask in enumerate(descriptor_mask):
    descriptor_mask[i] = mask.reshape((mask.shape[0], mask.shape[1], -1))

out = []
import sys
for i in range(len(tensors)):
    t, _ = model(tensors[i], descriptor_mask[i])
    out.append(t)

out = torch.stack(out, dim=1)
out = out.reshape((*out.shape[0:3], mask_size, mask_size))
out = out.squeeze().detach().cpu().numpy()
out = out.reshape((out.shape[0] * out.shape[1], *out.shape[2:]))

### plot
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 150.);

fig, ax = plt.subplots(1, figsize=(15,15))
ret, inp = cap.read()
frame = 0

width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

for i in range(100):
    im = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    plt.cla()
    ax.imshow(im)

    resized_mask = cv2.resize(out[frame, :, :], (int(width), int(height)), interpolation = cv2.INTER_AREA)
    print(np.sum(out[frame, :, :]))
    ax.imshow(resized_mask, cmap='jet', alpha=0.5, vmin=0, vmax=1)

    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.00001)
#    plt.savefig('images/file%05d.jpeg' % frame, bbox_inches = 'tight', pad_inches = 0)

    frame += 1
    ret, inp = cap.read()
plt.close()
#os.system("ffmpeg -framerate 20 -pattern_type glob -i 'images/*.jpeg' -c:v mpeg4 -vb 1M -qscale:v 2 " + "full-notnorm.mp4")











